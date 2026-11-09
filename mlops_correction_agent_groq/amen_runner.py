
import sys
from pathlib import Path

AMEN_ROOT = Path(__file__).resolve().parent / "amen_project"

if not AMEN_ROOT.is_dir():
    raise RuntimeError("Dossier amen_project introuvable.")

sys.path.insert(0, str(AMEN_ROOT))

from mlops_churn import investigate


def run_amen_investigation(
    episode="episode-003",
    provider="nvidia"
):
    base = AMEN_ROOT / "scenario_data" / "demo"
    episode_dir = base / "episodes" / episode

    if not episode_dir.is_dir():
        raise ValueError(
            f"Episode introuvable : {episode_dir}"
        )

    result = investigate(
        episode_dir=str(episode_dir),
        reference_dir=str(base),
        provider=provider,
        output_dir=str(
            Path("artifacts_v3") / "amen_results"
        ),
    )

    return result
