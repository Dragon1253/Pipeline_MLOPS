param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$platformPython = Join-Path $PSScriptRoot '.venv-cloud\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $platformPython)) { throw 'Environnement .venv-cloud introuvable. Consultez PLATFORM.md.' }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'frontend\dist\index.html'))) { throw 'Interface non compilée. Exécutez npm install puis npm run build dans frontend.' }
Write-Host "Sentinel : http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host 'Ctrl+C pour arrêter. Les investigations utilisent Nemotron via NVIDIA.'
& $platformPython -m uvicorn platform_api.app:app --host 127.0.0.1 --port $Port
