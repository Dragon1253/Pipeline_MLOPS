
"""Bounded LLM router for the post-analysis MLOps workflow."""

from pathlib import Path
from typing import Literal, TypedDict
import os
from urllib.parse import quote

from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph

from agent_v3 import CorrectionAgentV3, IncidentV3


def _extract_usage_metadata(value):
    if value is None:
        return {}

    def walk(node):
        usage = {}
        if isinstance(node, dict):
            lower_map = {str(k).lower(): v for k, v in node.items()}
            for key in ("input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens", "usage"):
                if key in lower_map:
                    usage[key] = lower_map[key]
            if "response_metadata" in lower_map:
                usage.update(walk(lower_map["response_metadata"]))
            if "usage_metadata" in lower_map:
                usage.update(walk(lower_map["usage_metadata"]))
            for nested in node.values():
                nested_usage = walk(nested)
                if nested_usage:
                    usage.update(nested_usage)
        elif isinstance(node, list):
            for item in node:
                nested_usage = walk(item)
                if nested_usage:
                    usage.update(nested_usage)
        return usage

    usage = walk(value)
    normalized = {}
    for key, val in usage.items():
        lower = str(key).lower()
        if lower in {"input_tokens", "prompt_tokens"}:
            normalized["input_tokens"] = val
        elif lower in {"output_tokens", "completion_tokens"}:
            normalized["output_tokens"] = val
        elif lower == "total_tokens":
            normalized["total_tokens"] = val
        elif lower == "usage":
            normalized.update(_extract_usage_metadata(val))
    return normalized


def load_local_groq_settings(env_path=None):
    """Load supported local settings from the adjacent .env without overriding the shell."""
    path = Path(env_path) if env_path else Path(__file__).with_name(".env")
    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        name, separator, value = entry.partition("=")
        name = name.strip()
        if not separator or name not in {
            "GROQ_API_KEY",
            "GROQ_MODEL",
            "SMTP_HOST",
            "SMTP_PORT",
            "SMTP_USERNAME",
            "SMTP_PASSWORD",
            "SMTP_FROM",
            "HUMAN_REVIEW_EMAIL",
            "HUMAN_REVIEW_BASE_URL",
            "HUMAN_REVIEW_DB",
        }:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value and not os.environ.get(name):
            os.environ[name] = value


# ============================================================
# 1. STRUCTURED DECISION
# ============================================================

class OrchestrationDecision(BaseModel):
    route: Literal["correction", "human_review"]
    reason: str = Field(min_length=8)


# ============================================================
# 2. LANGGRAPH STATE
# ============================================================

class OrchestrationState(TypedDict, total=False):
    incident: dict
    proposed_decision: dict
    decision: dict
    guard_result: dict
    correction_result: dict | None
    human_review: dict | None


# ============================================================
# 3. ORCHESTRATOR V4
# ============================================================

class OrchestratorAgentV4:
    """
    Bounded LLM orchestrator.

    The LLM proposes a route.
    The guard independently validates the route.
    The LLM cannot bypass safety rules.
    """

    def __init__(
        self,
        *,
        mode="rules",
        correction_mode="rules",
        out_dir="artifacts_v4",
        llm=None,
        enable_human_review_side_effects=True,
    ):

        if mode not in {"rules", "llm"}:
            raise ValueError("mode doit etre rules ou llm.")

        self.mode = mode
        self.correction_mode = correction_mode
        self.out_dir = Path(out_dir)
        self.llm = llm
        self.enable_human_review_side_effects = enable_human_review_side_effects
        load_local_groq_settings()

        # ----------------------------------------------------
        # GROQ CONFIGURATION
        # ----------------------------------------------------

        if mode == "llm" and self.llm is None:

            from langchain_groq import ChatGroq

            api_key = os.getenv("GROQ_API_KEY")

            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY absente. Renseigne-la dans le .env local ou dans l'environnement."
                )

            model_name = os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-120b"
            )

            self.llm = ChatGroq(
                model=model_name,
                temperature=0,
                max_retries=2,
            )

        # ----------------------------------------------------
        # LANGGRAPH WORKFLOW
        # ----------------------------------------------------

        graph = StateGraph(OrchestrationState)

        graph.add_node("decide", self._decide)
        graph.add_node("guard", self._guard)
        graph.add_node("correction", self._run_correction)
        graph.add_node("human_review", self._request_human_review)

        graph.add_edge(START, "decide")
        graph.add_edge("decide", "guard")

        graph.add_conditional_edges(
            "guard",
            self._next,
            {
                "correction": "correction",
                "human_review": "human_review",
            },
        )

        graph.add_edge("correction", END)
        graph.add_edge("human_review", END)

        self.graph = graph.compile()

    # ========================================================
    # 4. SAFETY ELIGIBILITY
    # ========================================================

    @staticmethod
    def _eligible(incident):

        validated = IncidentV3.model_validate(incident)
        details = validated.details

        return (
            validated.confidence >= 0.7
            and validated.type != "inconnu"
            and details.get("confidence_supported") is True
            and details.get("requires_human_review") is False
        )

    # ========================================================
    # 5. LLM DECISION
    # ========================================================

    def _decide(self, state: OrchestrationState):

        incident = IncidentV3.model_validate(
            state["incident"]
        )

        # ----------------------------------------------------
        # RULES MODE
        # ----------------------------------------------------

        if self.mode == "rules":

            route = (
                "correction"
                if self._eligible(incident.model_dump())
                else "human_review"
            )

            decision = OrchestrationDecision(
                route=route,
                reason="Routage determine par le score et les garde-fous.",
            )

        # ----------------------------------------------------
        # LLM MODE
        # ----------------------------------------------------

        else:

            prompt = (
                "Tu es l'orchestrateur d'un workflow MLOps.\n"
                "Tu dois choisir uniquement une route parmi : "
                "correction ou human_review.\n\n"

                "Regles obligatoires :\n"
                "- confidence >= 0.7\n"
                "- confidence_supported = true\n"
                "- requires_human_review = false\n"
                "- type different de inconnu\n\n"

                "Si toutes les conditions sont satisfaites, "
                "tu peux choisir correction.\n"

                "Sinon, choisis human_review.\n\n"

                "Tu ne peux pas modifier les donnees de l'incident.\n"
                "Tu ne choisis aucune action corrective.\n\n"

                "Retourne uniquement un objet JSON valide avec "
                "les champs route et reason.\n\n"

                "Format attendu :\n"
                '{"route":"human_review",'
                '"reason":"Explication de la decision"}\n\n'

                "Incident a analyser :\n"
                + incident.model_dump_json(indent=2)
            )

            try:

                structured = self.llm.with_structured_output(
                    OrchestrationDecision,
                    method="json_mode",
                    include_raw=True,
                ).invoke(prompt)

                raw_response = None
                if isinstance(structured, dict) and "parsed" in structured:
                    raw_response = structured.get("raw")
                    if structured.get("parsed") is None:
                        raise ValueError("La réponse LLM n'a pas pu être analysée.")
                    structured = structured["parsed"]

                if isinstance(structured, OrchestrationDecision):

                    decision = structured

                elif isinstance(structured, dict):

                    decision = OrchestrationDecision.model_validate(
                        structured
                    )

                else:

                    raise ValueError(
                        "Format de reponse LLM inattendu."
                    )

                usage = _extract_usage_metadata(getattr(raw_response, "response_metadata", None))
                if not usage:
                    usage = _extract_usage_metadata(getattr(raw_response, "usage_metadata", None))

            except Exception as e:

                print("\n========== ERREUR ORCHESTRATEUR GROQ ==========")
                print("Type :", type(e).__name__)
                print("Message :", str(e))
                print("================================================\n")

                raise

        decision_payload = decision.model_dump()
        if self.mode == "llm" and usage:
            decision_payload["usage"] = usage

        return {
            "proposed_decision": decision_payload,
            "mode": self.mode,
        }

    # ========================================================
    # 6. INDEPENDENT SAFETY GUARD
    # ========================================================

    def _guard(self, state: OrchestrationState):

        incident = IncidentV3.model_validate(
            state["incident"]
        )

        proposed = OrchestrationDecision.model_validate(
            state["proposed_decision"]
        )

        eligible = self._eligible(
            incident.model_dump()
        )

        # A correction is accepted only if eligible.
        # Human review is always an allowed route.

        accepted = (
            proposed.route == "human_review"
            or eligible
        )

        final_route = (
            proposed.route
            if accepted
            else "human_review"
        )

        reason = (
            proposed.reason
            if accepted
            else (
                "Garde-fou: score faible, type inconnu "
                "ou preuve de corroboration absente."
            )
        )

        return {

            "decision": {
                "route": final_route,
                "reason": reason,
            },

            "guard_result": {
                "accepted": accepted,
                "proposed_route": proposed.route,
                "final_route": final_route,
                "eligibility_check": eligible,
            },
        }

    # ========================================================
    # 7. ROUTING
    # ========================================================

    @staticmethod
    def _next(state: OrchestrationState):

        return state["decision"]["route"]

    # ========================================================
    # 8. CORRECTION AGENT
    # ========================================================

    def _run_correction(self, state: OrchestrationState):

        correction = CorrectionAgentV3(
            mode=self.correction_mode,
            out_dir=self.out_dir / "correction",
        ).run(state["incident"])

        return {
            "correction_result": correction,
            "human_review": None,
        }

    # ========================================================
    # 9. HUMAN REVIEW
    # ========================================================

    def _request_human_review(self, state: OrchestrationState):
        incident = state["incident"]
        details = incident.get("details", {})
        review_request = {
            "incident_id": incident["incident_id"],
            "incident_type": incident["type"],
            "confidence": incident["confidence"],
            "cause_probable": details.get("cause_probable", "inconnu"),
            "description": details.get("description", ""),
            "evidence_ids": details.get("evidence_ids", []),
            "reason": state["decision"]["reason"],
        }

        if not self.enable_human_review_side_effects:
            return {
                "correction_result": None,
                "human_review": {
                    "required": True,
                    "review_id": None,
                    "status": "simulated",
                    "reason": state["decision"]["reason"],
                    "request": review_request,
                    "notification": {"status": "suppressed"},
                    "production_modified": False,
                },
            }

        from human_notifications import send_human_review_email
        from human_review_store import HumanReviewStore, review_database_path

        created = HumanReviewStore(review_database_path()).create_review(
            request=review_request,
            incident=incident,
            correction_mode=self.correction_mode,
            run_dir=self.out_dir,
            pipeline_result_file=self.out_dir / "pipeline_result.json",
        )
        base_url = os.getenv(
            "HUMAN_REVIEW_BASE_URL", "http://127.0.0.1:5000"
        ).rstrip("/")
        review_url = (
            f"{base_url}/human-review/{created['review_id']}"
            f"?token={quote(created['token'])}"
        )
        notification = send_human_review_email({
            **review_request,
            "review_url": review_url,
        })

        return {
            "correction_result": None,
            "human_review": {
                "required": True,
                "review_id": created["review_id"],
                "status": "pending",
                "reason": state["decision"]["reason"],
                "request": review_request,
                "notification": notification,
                "production_modified": False,
            },
        }

    # ========================================================
    # 10. PIPELINE EXECUTION
    # ========================================================

    def run(self, incident):

        validated = IncidentV3.model_validate(
            incident
        )

        result = self.graph.invoke({
            "incident": validated.model_dump()
        })

        return result
