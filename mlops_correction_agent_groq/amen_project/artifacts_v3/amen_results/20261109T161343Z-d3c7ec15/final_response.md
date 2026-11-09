{
  "cause_probable": "concept_drift",
  "resume": "Alerte log_loss confirmée (0.54354 > seuil 0.5412) sur la fenêtre focus, isolée : les fenêtres de confirmation sont normales. Les données d'entrée sont intactes (source == inputs, 0 valeur modifiée), aucune dérige marginale significative des features (toutes q_value BH > 0.41), et le modèle servi est identique au modèle de référence (même hash). Par élimination de la corruption, du data drift et du changement de version, la dégradation de performance sur cette fenêtre est la plus vraisemblablement attribuable à un changement de la relation cible/feature (concept drift) sur cet épisode.",
  "preuves": ["E0001", "E0006", "E0008", "E0009", "E0010"]
}