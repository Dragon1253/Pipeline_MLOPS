"""Agent de correction: LangGraph + LLM optionnel + outils sûrs.
Mode rules = baseline déterministe; mode llm = planification par modèle de langage.
Aucune opération sur un vrai registre ou service de production.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict
import json
import os
from pydantic import BaseModel, Field
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Local offline fallback: same node order, not a substitute for LangGraph.
    LANGGRAPH_AVAILABLE = False
    START, END = "__start__", "__end__"
    class StateGraph:
        def __init__(self, schema):
            self.nodes = {}
            self.edges = {}
        def add_node(self, name, fn):
            self.nodes[name] = fn
        def add_edge(self, src, dst):
            self.edges[src] = dst
        def compile(self):
            nodes, edges = self.nodes, self.edges
            class LocalGraph:
                def invoke(self, state):
                    state = dict(state)
                    name = edges[START]
                    while name != END:
                        state.update(nodes[name](state))
                        name = edges[name]
                    return state
            return LocalGraph()

ACTIONS = {"close_alert", "quarantine_batch", "prepare_retraining", "prepare_rollback", "human_review"}

class IncidentV3(BaseModel):
    incident_id: str = Field(min_length=1)
    type: Literal["fausse_alerte", "derive_donnees", "qualite_donnees", "regression_version", "inconnu"]
    confidence: float = Field(ge=0, le=1)
    metrics: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)

class Plan(BaseModel):
    action: Literal["close_alert", "quarantine_batch", "prepare_retraining", "prepare_rollback", "human_review"]
    reason: str = Field(min_length=5)
    tool_args: dict[str, Any] = Field(default_factory=dict)

class State(TypedDict, total=False):
    incident: dict
    proposed_plan: dict
    plan: dict
    guard_result: dict
    mode: str
    tool_result: dict
    verification: dict
    history: list[dict]

class CorrectionAgentV3:
    def __init__(self, mode="rules", out_dir="artifacts_v3", llm=None):
        if mode not in {"rules", "llm"}:
            raise ValueError("mode doit être rules ou llm")
        self.mode = mode
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        if mode == "llm" and self.llm is None:
            from langchain_groq import ChatGroq
            if not os.getenv("GROQ_API_KEY"):
                raise RuntimeError("GROQ_API_KEY absente. Configure la variable dans PowerShell.")
            self.llm = ChatGroq(
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                temperature=0,
                max_retries=2,
            )
        graph = StateGraph(State)
        graph.add_node("plan", self._plan)
        graph.add_node("guard", self._guard)
        graph.add_node("execute", self._execute)
        graph.add_node("verify", self._verify)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "guard")
        graph.add_edge("guard", "execute")
        graph.add_edge("execute", "verify")
        graph.add_edge("verify", END)
        self.graph = graph.compile()

    @staticmethod
    def _expected(incident: IncidentV3) -> str:
        if incident.confidence < 0.7 or incident.type == "inconnu":
            return "human_review"
        return {"fausse_alerte": "close_alert", "qualite_donnees": "quarantine_batch",
                "derive_donnees": "prepare_retraining", "regression_version": "prepare_rollback"}[incident.type]







    def _plan(self, state: State) -> dict:
        inc = IncidentV3.model_validate(state["incident"])

        if self.mode == "rules":
            action = self._expected(inc)
            plan = Plan(
                action=action,
                reason=f"Baseline: diagnostic {inc.type}, confiance {inc.confidence:.2f}"
            )
        else:
            prompt = f"""
Tu es un agent intelligent de correction MLOps.

Ta mission est d'analyser un incident et de proposer
UNE action corrective adaptée.

ACTIONS AUTORISÉES :

1. close_alert
   Utiliser uniquement pour une fausse alerte confirmée,
   lorsque les performances restent stables.

2. quarantine_batch
   Utiliser lorsque les données sont corrompues,
   contiennent trop de valeurs manquantes ou sont suspectes.
   Ne jamais réentraîner sur des données corrompues.

3. prepare_retraining
   Utiliser lorsqu'une dérive réelle est confirmée,
   que les données sont valides et que les performances
   du modèle se dégradent.
   Préparer uniquement un candidat à évaluer.
   Aucun déploiement automatique.

4. prepare_rollback
   Utiliser lorsqu'une nouvelle version du modèle
   provoque une régression de performance.
   Comparer les versions avant toute décision.

5. human_review
   Utiliser si le diagnostic est inconnu,
   si la confiance est inférieure à 0.7
   ou si les informations sont insuffisantes.

RÈGLES IMPORTANTES :
- Une dérive de distribution n'est pas forcément
  une corruption des données.
- Ne choisis pas quarantine_batch simplement
  parce qu'un indicateur de dérive est élevé.
- Ne prétends jamais avoir exécuté un réentraînement,
  un rollback ou un déploiement.
- Justifie ton choix à partir des métriques disponibles.
- N'invente aucune mesure absente de l'incident.

INCIDENT :
{inc.model_dump_json(indent=2)}

Retourne une action, une raison et les arguments
nécessaires sous forme structurée.
"""
            plan = self.llm.with_structured_output(
                Plan,
                method="json_schema"
            ).invoke(prompt)

        return {
            "proposed_plan": plan.model_dump(),
            "plan": plan.model_dump(),
            "mode": self.mode
        }








    def _guard(self, state: State) -> dict:
        inc = IncidentV3.model_validate(state["incident"])
        proposed = Plan.model_validate(state["proposed_plan"])

        expected = self._expected(inc)

        if proposed.action != expected:
            safe_plan = Plan(
                action="human_review",
                reason=(
                    f"Plan refusé par garde-fou : "
                    f"{proposed.action}; attendu : {expected}"
                )
            )

            return {
                "plan": safe_plan.model_dump(),
                "guard_result": {
                    "accepted": False,
                    "proposed_action": proposed.action,
                    "expected_action": expected,
                    "final_action": "human_review"
                }
            }

        return {
            "guard_result": {
                "accepted": True,
                "proposed_action": proposed.action,
                "expected_action": expected,
                "final_action": proposed.action
            }
        }


























    def _execute(self, state: State) -> dict:
        inc = IncidentV3.model_validate(state["incident"])
        plan = Plan.model_validate(state["plan"])
        action = plan.action

        if action == "quarantine_batch":
            batch_id = str(
                inc.details.get("batch_id") or inc.incident_id
            )
            result = {
                "status": "simulated",
                "batch_id": batch_id,
                "blocked_for_retraining": True,
            }

        elif action == "close_alert":
            result = {
                "status": "simulated",
                "alert_closed": True,
            }

        elif action == "prepare_retraining":
            result = {
                "status": "pending_evaluation",
                "candidate_deployed": False,
                "requires_evaluation": True,
                "requires_human_approval": True,
            }

        elif action == "prepare_rollback":
            # Exécuter la comparaison réelle uniquement
            # pour les incidents explicitement liés à la démo.
            if inc.details.get("run_local_rollback_demo") is True:
                try:
                    from rollback_tool import prepare_rollback
                    report = prepare_rollback()

                    result = {
                        "status": "pending_evaluation",
                        "rollback_recommended": report[
                            "rollback_recommended"
                        ],
                        "current_mae": report["current_mae"],
                        "previous_mae": report["previous_mae"],
                        "rollback_executed": False,
                        "requires_human_approval": True,
                    }
                except Exception as exc:
                    result = {
                        "status": "pending_human_review",
                        "error": str(exc),
                        "rollback_executed": False,
                        "production_modified": False,
                    }
            else:
                result = {
                    "status": "pending_evaluation",
                    "rollback_executed": False,
                    "requires_version_comparison": True,
                    "requires_human_approval": True,
                }

        else:
            result = {
                "status": "pending_human_review",
                "production_modified": False,
            }

        return {"tool_result": result}














    def _verify(self, state: State) -> dict:
        plan = Plan.model_validate(state["plan"])
        result = state["tool_result"]
        safe = (plan.action in ACTIONS and result.get("status") in
                {"simulated", "pending_evaluation", "pending_human_review"}
                and result.get("candidate_deployed") is not True
                and result.get("rollback_executed") is not True)
        report = {
            "incident_id": state["incident"]["incident_id"],
            "mode": self.mode,
            "llm_proposed_action": state["proposed_plan"]["action"],
            "final_action": state["plan"]["action"],
            "proposed_plan": state["proposed_plan"],
            "plan": state["plan"],
            "guard_result": state["guard_result"],
            "tool_result": result,
            "verification": {
                "safe": safe,
                "production_unchanged": True
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        with (self.out / "agent_traces.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")
        return {"verification": report["verification"], "history": [report]}

    def run(self, incident: IncidentV3 | dict) -> dict:
        data = IncidentV3.model_validate(incident).model_dump()
        result = self.graph.invoke({"incident": data})
        return result["history"][-1]
