# Agent de correction MLOps — Groq + Llama

## Installation (PowerShell dans VS Code)

```powershell
python -m pip install -r requirements.txt
$env:GROQ_API_KEY = Read-Host "Cle API Groq"
python demo_v3.py --mode rules
python -m unittest -v test_v3.py
python demo_v3.py --mode llm
```

Clé à créer sur https://console.groq.com/keys. Ne jamais la committer ni la partager.
Modèle par défaut : `llama-3.3-70b-versatile`, personnalisable via `GROQ_MODEL`.

## Fonctionnement

Le workflow LangGraph suit plan → guard → execute → verify. En mode `llm`, Groq propose un plan structuré Pydantic. Le garde-fou déterministe refuse une action incompatible avec le diagnostic et demande une revue humaine. Les outils sont **simulés** : pas de vrai réentraînement, rollback ou modification de production.

Le mode `rules` sert de baseline et ne contacte pas Groq. Les tests automatiques sont hors-ligne.
