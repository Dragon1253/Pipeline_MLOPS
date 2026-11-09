param(
    [ValidateSet("data_drift", "concept_drift", "data_corruption", "deployment_regression", "normal")]
    [string]$Scenario = "data_corruption"
)

$ErrorActionPreference = "Stop"
$projectDirectory = $PSScriptRoot
$pythonExe = Join-Path (Split-Path -Parent $projectDirectory) ".venv-correction\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python du venv introuvable : $pythonExe"
}
if (-not (Test-Path -LiteralPath (Join-Path $projectDirectory ".env"))) {
    throw "Fichier .env introuvable dans $projectDirectory. Copie .env.example vers .env et ajoute ta clé Groq."
}

$previousLocation = Get-Location
$pipelineExitCode = 1
try {
    Remove-Item Env:GROQ_API_KEY, Env:GROQ_MODEL -ErrorAction SilentlyContinue
    Set-Location -LiteralPath $projectDirectory
    & $pythonExe pipeline_v4.py --scenario $Scenario --analysis-mode fixture --orchestrator-mode llm
    $pipelineExitCode = $LASTEXITCODE
}
finally {
    Set-Location -LiteralPath $previousLocation
    Remove-Item Env:GROQ_API_KEY, Env:GROQ_MODEL -ErrorAction SilentlyContinue
}

if ($pipelineExitCode -ne 0) {
    throw "Le pipeline s'est terminé avec le code $pipelineExitCode."
}