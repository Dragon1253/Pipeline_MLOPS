"""Commandes du monitoring.

    python -m monitoring run --scenario data_corruption --seed 1
    python -m monitoring run --scenario deployment_regression --seed 2 --investigate nvidia
    python -m monitoring evaluate --seeds 30
    python -m monitoring dataset --split train --workers 4
    python -m monitoring predict --target target_h2 --budget 5
    python -m monitoring predict --target target_h2 --sequence gru lstm
    python -m monitoring analyze --target target_h2
    python -m monitoring train-predictor
    python -m monitoring run --scenario data_corruption --seed 7 --label-delay 2 --predict
"""
import argparse
import json
import os
import warnings

# joblib cherche le nombre de cœurs physiques avec « wmic », absent des Windows récents, et
# affiche un avertissement sans conséquence. Une limite inférieure au nombre total de cœurs
# lui évite cette recherche (une limite égale ne suffit pas).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))

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
    run.add_argument("--predict", action="store_true",
                     help="ajouter le risque prédit et les alertes prédictives (prédicteur entraîné requis)")
    run.add_argument("--predictor", help="chemin du prédicteur (par défaut monitoring_runs/predictor/predictor.joblib)")
    evaluate = sub.choices["evaluate"]
    evaluate.add_argument("--scenarios", nargs="+", choices=STREAM_SCENARIOS, default=list(STREAM_SCENARIOS))
    evaluate.add_argument("--seeds", type=int, default=30, help="nombre de flux par scénario")
    evaluate.add_argument("--first-seed", type=int, default=1)
    dataset = sub.add_parser("dataset", help="générer le jeu de données de prédiction des alertes")
    dataset.add_argument("--split", nargs="+", default=["train", "validation", "test", "generalization"],
                         choices=["train", "validation", "test", "generalization"])
    dataset.add_argument("--streams", type=int, help="nombre de flux (par défaut : taille prévue du découpage)")
    dataset.add_argument("--workers", type=int, default=1, help="processus en parallèle")
    predict = sub.add_parser("predict", help="entraîner et comparer les modèles de prédiction de l'alerte")
    predict.add_argument("--target", default="target_now",
                         choices=["target_now", "target_h1", "target_h2", "target_h3"])
    predict.add_argument("--budget", type=float, default=5.0,
                         help="fausses alarmes autorisées pour 100 fenêtres saines (choix du seuil)")
    predict.add_argument("--groups", nargs="+", default=["sig", "known", "trend"],
                         choices=["sig", "known", "trend"], help="groupes de signaux utilisés (ablation)")
    predict.add_argument("--sequence", nargs="*", default=[], choices=["gru", "lstm"],
                         help="ajouter les modèles séquentiels (PyTorch requis)")
    predict.add_argument("--epochs", type=int, default=30, help="époques maximales des modèles séquentiels")
    analyze = sub.add_parser("analyze", help="ablation, intervalles de confiance et explication des alertes (M2)")
    analyze.add_argument("--target", default="target_h2",
                         choices=["target_now", "target_h1", "target_h2", "target_h3"])
    analyze.add_argument("--budget", type=float, default=5.0)
    analyze.add_argument("--bootstrap", type=int, default=1000, help="tirages du bootstrap")
    analyze.add_argument("--no-ablation", action="store_true", help="sauter l'ablation (plus rapide)")
    trainer = sub.add_parser("train-predictor", help="entraîner et enregistrer le prédicteur utilisé en direct")
    trainer.add_argument("--target", default="target_h2",
                         choices=["target_now", "target_h1", "target_h2", "target_h3"])
    trainer.add_argument("--budget", type=float, default=5.0)
    return parser


def main(argv=None):
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
    args = build_parser().parse_args(argv)
    if args.command == "dataset":
        from .dataset import build_split
        for split in args.split:
            def progress(done, total, split=split):
                if done % 50 == 0 or done == total:
                    print(f"  {split} : {done}/{total} flux", flush=True)
            path = build_split(split, n_streams=args.streams, workers=args.workers, progress=progress)
            manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            print(f"{split} : {manifest['rows']} lignes, {manifest['streams']} flux -> {path}")
            print("  part des cibles à 1 : " + ", ".join(f"{k} {v:.1%}" for k, v in manifest["positive_rate"].items()))
        return 0
    if args.command == "train-predictor":
        from .predictor import train_predictor
        path, info = train_predictor(args.target, args.budget)
        test = info["test"]
        print(f"Prédicteur enregistré : {path}")
        print(f"  cible {info['target']} · seuil d'alarme {info['threshold']:.3f} · {info['n_features']} entrées")
        print(f"  test : détection {100 * test['detection_rate']:.0f} % · avance sur R1 {test['mean_lead_vs_reactive']:.2f} "
              f"fenêtre · fausses alarmes {test['false_alarms_per_100_clean_windows']:.1f} / 100")
        return 0
    if args.command == "analyze":
        from .analysis import run as analyze
        folder, _ = analyze(args.target, args.budget, n_boot=args.bootstrap, ablate=not args.no_ablation)
        print((folder / "report.md").read_text(encoding="utf-8"))
        print(f"Résultats : {folder}")
        return 0
    if args.command == "predict":
        from .prediction import run
        folder, _ = run(args.target, args.budget, groups=tuple(args.groups),
                        sequence_models=tuple(args.sequence), sequence_epochs=args.epochs)
        print((folder / "report.md").read_text(encoding="utf-8"))
        print(f"Résultats : {folder}")
        return 0
    common = dict(n_windows=args.windows, onset=args.onset, ramp=args.ramp, max_intensity=args.intensity)
    if args.command == "run":
        from .service import run_stream
        predictor = None
        if args.predict or args.predictor:
            from .predictor import DEFAULT_PREDICTOR, load_predictor
            predictor = load_predictor(args.predictor or DEFAULT_PREDICTOR)
        summary = run_stream(StreamSpec(args.scenario, args.seed, **common),
                             label_delay=args.label_delay, persistence=args.persistence, predictor=predictor)
        confirmed = {a["window_id"]: a for a in summary["alerts"]}
        predicted = {a["window_id"]: a for a in summary["predictive_alerts"]}
        risks = {r["window"]: r["risk"] for r in summary["risk_timeline"]}
        for row in summary["timeline"]:
            window = row["window"]
            loss = "en attente" if row["log_loss"] is None else f"{row['log_loss']:.3f}"
            marks = []
            if window in predicted:
                marks.append(f"ALERTE PRÉDICTIVE (pas {predicted[window]['detected_at_step']})")
            if window in confirmed:
                marks.append(f"ALERTE CONFIRMÉE {confirmed[window]['severity']} (pas {confirmed[window]['detected_at_step']})")
            risk = f"  risque {risks[window]:.2f}" if window in risks else ""
            print(f"{window}  log loss {loss:>10}{risk}  {' · '.join(marks)}")
        print(json.dumps({k: summary[k] for k in ("run_dir", "first_alert_window", "first_alert_step", "episodes")},
                         ensure_ascii=False, indent=2))
        for alert in summary["predictive_alerts"]:
            signals = ", ".join(f"{s['signal']} = {s['value']:.3f} (habituel {s['usual_value']:.3f})"
                                for s in alert["top_signals"] if s["value"] is not None)
            lead = f", {alert['lead_steps']} pas avant l'alerte confirmée" if alert.get("lead_steps") is not None else ""
            print(f"Alerte prédictive {alert['window_id']} : risque {alert['risk']:.2f}, statut {alert['status']}{lead}")
            print(f"  signaux : {signals}")
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
