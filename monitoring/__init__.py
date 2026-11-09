"""Monitoring du modèle de churn : flux, métriques, alertes et paquets pour l'agent d'analyse."""
import os

# joblib cherche le nombre de cœurs physiques avec « wmic », absent des Windows récents, et
# affiche un avertissement sans conséquence. Une limite inférieure au nombre total de cœurs
# lui évite cette recherche (une limite égale ne suffit pas). Placé ici pour couvrir la
# ligne de commande comme les tests.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))
