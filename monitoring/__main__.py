"""Commandes du monitoring.

    python -m monitoring run --scenario data_corruption --seed 1
    python -m monitoring run --scenario deployment_regression --seed 2 --investigate nvidia
    python -m monitoring evaluate --seeds 30
"""
import argparse
import json
import warnings

from .stream import STREAM_SCENARIOS, StreamSpec


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m monitoring", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--windows", type=int, default=20, help="nombre de fenêtres du flux")
        p.add_argument("--onset", type=int, default=8, help="indice de la première fenêtre touchée")
        p.add_argument("--ramp", type=int, default=4, help="fenêtres de montée (0 = incident brutal)")
        p.add_argument("--intensity", type=float, default=1.0, help="intensité maximale, entre 0 et 1")
        p.add_argument("--label-delay", type=int, default=1, help="retard des labels, en fenêtres")
        p.add_argument("--persistence", type=int, default=1, help="fenêtres consécutives au-dessus du seuil")
    run = sub.choices["run"]
    run.add_argument("--scenario", choices=STREAM_SCENARIOS, required=True)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--investigate", choices=("nvidia", "local"),
                     help="transmettre le premier paquet d'alerte à l'agent d'analyse")
    evaluate = sub.choices["evaluate"]
    evaluate.add_argument("--scenarios", nargs="+", choices=STREAM_SCENARIOS, default=list(STREAM_SCENARIOS))
    evaluate.add_argument("--seeds", type=int, default=30, help="nombre de flux par scénario")
    evaluate.add_argument("--first-seed", type=int, default=1)
    return parser


def main(argv=None):
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
    args = build_parser().parse_args(argv)
    common = dict(n_windows=args.windows, onset=args.onset, ramp=args.ramp, max_intensity=args.intensity)
    if args.command == "run":
        from .service import run_stream
        summary = run_stream(StreamSpec(args.scenario, args.seed, **common),
                             label_delay=args.label_delay, persistence=args.persistence)
        severity = {a["window_id"]: a["severity"] for a in summary["alerts"]}
        for row in summary["timeline"]:
            mark = f"ALERTE ({severity[row['window']]})" if row["window"] in severity else ""
            loss = "en attente" if row["log_loss"] is None else f"{row['log_loss']:.3f}"
            print(f"{row['window']}  log loss {loss:>10}  {mark}")
        print(json.dumps({k: summary[k] for k in ("run_dir", "first_alert_window", "first_alert_step", "episodes")},
                         ensure_ascii=False, indent=2))
        if args.investigate and summary["episodes"]:
            from mlops_churn.agent import investigate
            from .service import DEFAULT_REFERENCE
            result = investigate(summary["episodes"][-1], DEFAULT_REFERENCE, provider=args.investigate,
                                 output_dir="results")
            print(json.dumps({"status": result["status"], "diagnosis": result["final_diagnosis"],
                              "run_directory": result["run_directory"]}, ensure_ascii=False, indent=2))
        return 0
    from .evaluate import evaluate, markdown, save
    result = evaluate(args.scenarios, range(args.first_seed, args.first_seed + args.seeds),
                      label_delay=args.label_delay, persistence=args.persistence, **common)
    folder = save(result)
    print(markdown(result))
    print(f"Résultats : {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
