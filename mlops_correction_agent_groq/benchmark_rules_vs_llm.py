"""Benchmark of the deterministic rules router vs the LLM router across seeds.

The benchmark uses the real orchestrator decision path but keeps the synthetic incident
payload controlled by a seed so the result is reproducible across runs. A fallback
LLM stub is used only when no Groq API key is configured, so the harness still works
in local/offline execution without exposing or requiring secrets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from orchestrator_v4 import OrchestratorAgentV4, load_local_groq_settings

ALLOWED_INCIDENT_TYPES = (
    "fausse_alerte",
    "qualite_donnees",
    "derive_donnees",
    "regression_version",
    "inconnu",
)
DEFAULT_SCENARIOS = ("standard", "near_threshold", "hard_case")

MODEL_PRICE_USD_PER_1M = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.75},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "deepseek-r1-distill-llama-70b": {"input": 0.75, "output": 0.99},
}


def _scenario_seed(seed: int, incident_type: str, scenario: str) -> int:
    values = {name: index for index, name in enumerate(ALLOWED_INCIDENT_TYPES)}
    offsets = {"standard": 0, "near_threshold": 17, "hard_case": 91}
    return int(seed * 1000 + values.get(incident_type, 0) * 97 + offsets.get(scenario, 0) * 13 + 17)


def _extract_usage_metadata(payload: Any) -> dict:
    if payload is None:
        return {}

    def walk(value):
        usage = {}
        if isinstance(value, dict):
            lower_map = {str(k).lower(): v for k, v in value.items()}
            for key in ("input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens", "usage"):
                if key in lower_map:
                    usage[key] = lower_map[key]
            if "response_metadata" in lower_map:
                usage.update(walk(lower_map["response_metadata"]))
            if "usage_metadata" in lower_map:
                usage.update(walk(lower_map["usage_metadata"]))
            for nested in value.values():
                nested_usage = walk(nested)
                if nested_usage:
                    usage.update(nested_usage)
        elif isinstance(value, list):
            for item in value:
                nested_usage = walk(item)
                if nested_usage:
                    usage.update(nested_usage)
        return usage

    usage = walk(payload)
    normalized = {}

    for key, value in usage.items():
        normalized_key = str(key).lower()
        if normalized_key in {"input_tokens", "prompt_tokens"}:
            normalized["input_tokens"] = value
        elif normalized_key in {"output_tokens", "completion_tokens"}:
            normalized["output_tokens"] = value
        elif normalized_key == "total_tokens":
            normalized["total_tokens"] = value
        elif normalized_key == "usage":
            nested = _extract_usage_metadata(value)
            normalized.update(nested)

    return normalized


def _estimate_groq_cost_usd(usage: dict | None, model_name: str | None = None) -> float:
    if not usage:
        return 0.0

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens", 0)
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens", 0)

    try:
        input_tokens = int(input_tokens)
    except (TypeError, ValueError):
        input_tokens = 0
    try:
        output_tokens = int(output_tokens)
    except (TypeError, ValueError):
        output_tokens = 0

    if input_tokens == 0 and output_tokens == 0:
        return 0.0

    model = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    pricing = MODEL_PRICE_USD_PER_1M.get(model, MODEL_PRICE_USD_PER_1M["openai/gpt-oss-120b"])
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 8)


def _candidate_scenarios() -> tuple[str, ...]:
    return DEFAULT_SCENARIOS


class RuleAwareStructuredLLM:
    """Small offline stand-in for the LLM decision API.

    It follows the same gate as the rules router but exposes the same
    `with_structured_output(...).invoke()` contract expected by the orchestrator.
    """

    def __init__(self, incident: dict):
        self.incident = incident

    def with_structured_output(self, schema, **_kwargs):
        route = _expected_route(self.incident)

        class StructuredOutput:
            def invoke(self, _prompt):
                return schema(route=route, reason="Route decidee par le fallback local des tests.")

        return StructuredOutput()


def _expected_route(incident: dict) -> Literal["correction", "human_review"]:
    details = incident.get("details", {})
    confidence = float(incident.get("confidence", 0.0))
    incident_type = incident.get("type")
    eligible = (
        confidence >= 0.7
        and incident_type != "inconnu"
        and bool(details.get("confidence_supported"))
        and not bool(details.get("requires_human_review"))
    )
    return "correction" if eligible else "human_review"


def make_seeded_incident(seed: int, incident_type: str, scenario: str = "standard") -> dict:
    """Create a deterministic incident near the routing threshold."""
    if incident_type not in ALLOWED_INCIDENT_TYPES:
        raise ValueError(f"Type inconnu: {incident_type!r}")
    if scenario not in _candidate_scenarios():
        raise ValueError(f"Scénario inconnu: {scenario!r}; attendu {list(_candidate_scenarios())}.")

    rng = random.Random(_scenario_seed(seed, incident_type, scenario))

    if scenario == "near_threshold":
        confidence = 0.7 + (rng.random() - 0.5) * 0.08
        confidence = max(0.52, min(0.88, confidence))
    elif scenario == "hard_case":
        confidence = 0.68 + rng.random() * 0.12
        confidence = max(0.52, min(0.9, confidence))
    else:
        confidence = 0.78 + (rng.random() - 0.5) * 0.22
        confidence = max(0.1, min(0.98, confidence))

    if incident_type == "inconnu":
        supported = False
        requires_human_review = True
    else:
        supported = True
        requires_human_review = bool((seed + len(incident_type)) % 3 == 0)

    if scenario == "hard_case":
        supported = bool((seed + len(incident_type)) % 2 == 0)
        requires_human_review = bool((seed + len(incident_type)) % 2 == 1)
        confidence = min(confidence, 0.72) if supported else max(confidence, 0.71)

    if scenario == "near_threshold":
        requires_human_review = bool((seed + len(incident_type)) % 4 == 0)
        if not supported:
            confidence = min(confidence, 0.68)

    if incident_type == "fausse_alerte":
        confidence = 0.83 + (seed % 5) * 0.02
    elif incident_type == "qualite_donnees":
        confidence = 0.74 + (seed % 4) * 0.03
    elif incident_type == "derive_donnees":
        confidence = 0.8 + (seed % 6) * 0.025
    elif incident_type == "regression_version":
        confidence = 0.76 + (seed % 5) * 0.03

    if seed % 2 == 0 and incident_type != "inconnu":
        confidence = min(0.68, confidence)

    if incident_type == "inconnu":
        confidence = min(confidence, 0.65)

    if scenario == "near_threshold":
        confidence = round(0.7 + ((seed % 5) - 2) * 0.01, 4)
        if incident_type == "inconnu":
            confidence = 0.64
    elif scenario == "hard_case":
        confidence = round(0.69 + (seed % 3) * 0.01, 4)
        if incident_type == "inconnu":
            confidence = 0.64

    details = {
        "batch_id": f"batch-{seed}-{incident_type}",
        "confidence_supported": supported,
        "requires_human_review": requires_human_review,
        "description": f"Incident {incident_type} generate aléatoirement avec une graine {seed}.",
        "evidence_ids": [f"E-{seed}-{index}" for index in range(1, 4)],
        "scenario": scenario,
    }

    return {
        "incident_id": f"BENCH-{incident_type.upper()}-{seed}",
        "type": incident_type,
        "confidence": round(confidence, 4),
        "metrics": {
            "seed": seed,
            "magnitude": round(rng.random() * 100, 2),
            "scenario": scenario,
        },
        "details": details,
    }


def _make_llm_for_mode(mode: str, incident: dict, llm_factory: Callable[[dict], object] | None):
    if mode != "llm":
        return None
    if llm_factory is not None:
        return llm_factory(incident)

    load_local_groq_settings()

    try:
        import langchain_groq  # noqa: F401
    except Exception:
        return RuleAwareStructuredLLM(incident)

    if os.getenv("GROQ_API_KEY"):
        return None
    return RuleAwareStructuredLLM(incident)


def _is_rate_limit_error(error: Exception) -> bool:
    response = getattr(error, "response", None)
    return (
        type(error).__name__ == "RateLimitError"
        or getattr(error, "status_code", None) == 429
        or getattr(response, "status_code", None) == 429
    )


def benchmark_rules_vs_llm(
    *,
    seeds: Iterable[int] = (1, 2, 3, 4, 5),
    incident_types: Iterable[str] = ALLOWED_INCIDENT_TYPES,
    modes: Iterable[str] = ("rules", "llm"),
    scenarios: Iterable[str] = DEFAULT_SCENARIOS,
    llm_factory: Callable[[dict], object] | None = None,
    llm_delay_seconds: float = 10.0,
    show_progress: bool = False,
    output_dir: str | Path | None = "artifacts_v4/benchmark",
) -> dict:
    """Return a structured comparison between the rules and llm routers."""

    seed_list = list(seeds)
    incident_list = list(incident_types)
    mode_list = list(modes)
    scenario_list = list(scenarios)
    if llm_delay_seconds < 0:
        raise ValueError("llm_delay_seconds doit être positif ou nul.")

    records = []
    last_remote_llm_call = None
    stop_reason = None
    total_runs = len(mode_list) * len(seed_list) * len(incident_list) * len(scenario_list)
    completed_runs = 0
    if show_progress:
        print(f"Benchmark démarré: {total_runs} exécutions prévues.", flush=True)
    totals = {
        "runs": 0,
        "correct_routes": 0,
        "wrong_routes": 0,
        "false_corrections": 0,
        "missed_human_reviews": 0,
        "total_latency_ms": 0.0,
        "total_cost_usd": 0.0,
        "avg_latency_ms": 0.0,
        "avg_cost_usd_per_run": 0.0,
    }
    mode_summary = {mode: {
        "runs": 0,
        "correct_routes": 0,
        "wrong_routes": 0,
        "false_corrections": 0,
        "missed_human_reviews": 0,
        "total_latency_ms": 0.0,
        "total_cost_usd": 0.0,
        "avg_latency_ms": 0.0,
        "avg_cost_usd_per_run": 0.0,
    } for mode in mode_list}

    for mode in mode_list:
        if mode not in {"rules", "llm"}:
            raise ValueError(f"Mode inconnu: {mode!r} ; attendu 'rules' ou 'llm'.")

        for seed in seed_list:
            for incident_type in incident_list:
                for scenario in scenario_list:
                    incident = make_seeded_incident(seed, incident_type, scenario=scenario)
                    expected_route = _expected_route(incident)
                    llm_instance = _make_llm_for_mode(mode, incident, llm_factory)

                    run_number = completed_runs + 1
                    if show_progress:
                        print(
                            f"[{run_number}/{total_runs}] {mode} | seed={seed} | "
                            f"{incident_type} | {scenario} : en cours",
                            flush=True,
                        )

                    agent = OrchestratorAgentV4(
                        mode=mode,
                        llm=llm_instance,
                        out_dir=str(output_dir or Path("artifacts_v4") / "benchmark" / mode),
                        enable_human_review_side_effects=False,
                    )
                    if mode == "llm" and llm_instance is None and last_remote_llm_call is not None:
                        elapsed = time.monotonic() - last_remote_llm_call
                        if elapsed < llm_delay_seconds:
                            if show_progress:
                                print(
                                    f"  Pause anti-quota: {llm_delay_seconds - elapsed:.1f} s",
                                    flush=True,
                                )
                            time.sleep(llm_delay_seconds - elapsed)
                    if mode == "llm" and llm_instance is None:
                        last_remote_llm_call = time.monotonic()

                    start_time = time.perf_counter()
                    try:
                        result = agent.run(incident)
                    except Exception as error:
                        if not _is_rate_limit_error(error):
                            raise
                        stop_reason = str(error)
                        if show_progress:
                            print(
                                f"  Quota Groq atteinte après {completed_runs} cas; "
                                "export des résultats déjà terminés.",
                                flush=True,
                            )
                        break

                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    usage_payload = result.get("proposed_decision", {}).get("usage")
                    if usage_payload is None:
                        usage_payload = result.get("decision", {}).get("usage")
                    actual_route = result["decision"]["route"]
                    cost_usd = _estimate_groq_cost_usd(
                        _extract_usage_metadata(usage_payload),
                        os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                    )
                    is_correct = actual_route == expected_route
                    is_false_correction = actual_route == "correction" and expected_route == "human_review"
                    is_missed_human_review = actual_route == "human_review" and expected_route == "correction"

                    record = {
                        "seed": seed,
                        "incident_type": incident_type,
                        "scenario": scenario,
                        "mode": mode,
                        "expected_route": expected_route,
                        "actual_route": actual_route,
                        "correct": is_correct,
                        "false_correction": is_false_correction,
                        "missed_human_review": is_missed_human_review,
                        "confidence": incident["confidence"],
                        "confidence_supported": incident["details"]["confidence_supported"],
                        "requires_human_review": incident["details"]["requires_human_review"],
                        "latency_ms": round(latency_ms, 2),
                        "cost_usd": round(cost_usd, 8),
                    }
                    records.append(record)

                    totals["runs"] += 1
                    totals["correct_routes"] += int(is_correct)
                    totals["wrong_routes"] += int(not is_correct)
                    totals["false_corrections"] += int(is_false_correction)
                    totals["missed_human_reviews"] += int(is_missed_human_review)
                    totals["total_latency_ms"] += latency_ms
                    totals["total_cost_usd"] += cost_usd

                    mode_summary[mode]["runs"] += 1
                    mode_summary[mode]["correct_routes"] += int(is_correct)
                    mode_summary[mode]["wrong_routes"] += int(not is_correct)
                    mode_summary[mode]["false_corrections"] += int(is_false_correction)
                    mode_summary[mode]["missed_human_reviews"] += int(is_missed_human_review)
                    mode_summary[mode]["total_latency_ms"] += latency_ms
                    mode_summary[mode]["total_cost_usd"] += cost_usd
                    completed_runs += 1
                    if show_progress:
                        print(
                            f"  terminé: route={actual_route}, correct={is_correct}, "
                            f"latence={latency_ms:.0f} ms, coût=${cost_usd:.8f}",
                            flush=True,
                        )
                if stop_reason:
                    break
            if stop_reason:
                break
        if stop_reason:
            break

    totals["avg_latency_ms"] = round(totals["total_latency_ms"] / max(1, totals["runs"]), 2)
    totals["avg_cost_usd_per_run"] = round(totals["total_cost_usd"] / max(1, totals["runs"]), 8)
    for mode in mode_list:
        mode_runs = mode_summary[mode]["runs"]
        mode_summary[mode]["avg_latency_ms"] = round(
            mode_summary[mode]["total_latency_ms"] / max(1, mode_runs),
            2,
        )
        mode_summary[mode]["avg_cost_usd_per_run"] = round(
            mode_summary[mode]["total_cost_usd"] / max(1, mode_runs),
            8,
        )

    summary = {
        "status": "partial" if stop_reason else "complete",
        "modes": mode_list,
        "seeds": seed_list,
        "incident_types": incident_list,
        "scenarios": scenario_list,
        "totals": totals,
        "mode_summary": mode_summary,
        "records": records,
        "accuracy": {
            mode: round(
                sum(
                    1 for record in records if record["mode"] == mode and record["correct"]
                ) / max(1, sum(1 for record in records if record["mode"] == mode)),
                4,
            )
            for mode in mode_list
        },
        "false_correction_rate": {
            mode: round(
                sum(
                    1 for record in records if record["mode"] == mode and record["false_correction"]
                ) / max(1, sum(1 for record in records if record["mode"] == mode)),
                4,
            )
            for mode in mode_list
        },
    }
    if stop_reason:
        summary["stop_reason"] = stop_reason

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_path = out_path / "benchmark_rules_vs_llm.json"
        csv_path = out_path / "benchmark_rules_vs_llm.csv"
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(records[0].keys()) if records else [
                "seed",
                "incident_type",
                "scenario",
                "mode",
                "expected_route",
                "actual_route",
                "correct",
                "false_correction",
                "missed_human_review",
                "confidence",
                "confidence_supported",
                "requires_human_review",
                "latency_ms",
                "cost_usd",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        summary["report_path"] = str(report_path)
        summary["csv_path"] = str(csv_path)

    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--incident-types",
        nargs="+",
        default=list(ALLOWED_INCIDENT_TYPES),
        help="Types d'incident a comparer pour le benchmark.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(DEFAULT_SCENARIOS),
        choices=DEFAULT_SCENARIOS,
        help="Scenarios de benchmark: standard, near_threshold, hard_case.",
    )
    parser.add_argument("--modes", choices=("rules", "llm"), nargs="+", default=["rules", "llm"])
    parser.add_argument(
        "--llm-delay-seconds",
        type=float,
        default=10.0,
        help="Pause minimale entre les appels Groq réels pour limiter les erreurs 429 (défaut: 10 s).",
    )
    parser.add_argument("--output-dir", default="artifacts_v4/benchmark")
    args = parser.parse_args(argv)

    summary = benchmark_rules_vs_llm(
        seeds=args.seeds,
        incident_types=args.incident_types,
        modes=args.modes,
        scenarios=args.scenarios,
        llm_delay_seconds=args.llm_delay_seconds,
        show_progress=True,
        output_dir=args.output_dir,
    )

    print(json.dumps({
        "modes": summary["modes"],
        "totals": summary["totals"],
        "accuracy": summary["accuracy"],
        "false_correction_rate": summary["false_correction_rate"],
        "status": summary["status"],
        "report_path": summary.get("report_path"),
        "csv_path": summary.get("csv_path"),
        "stop_reason": summary.get("stop_reason"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
