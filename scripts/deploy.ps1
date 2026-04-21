# Deploy script: precompute → git push → Vercel deploy
# Run from repo root: .\scripts\deploy.ps1
param(
    [switch]$SkipPrecompute,
    [switch]$SkipGit,
    [switch]$SkipVercel,
    [string]$RepoName = "av-curation-pipeline"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not $SkipPrecompute) {
    Write-Host "==> running precompute pipeline" -ForegroundColor Cyan
    python -m precompute.run --source synthetic --n 400
}

if (-not $SkipGit) {
    Write-Host "==> committing & pushing" -ForegroundColor Cyan
    git add -A
    if (git status --porcelain) { git commit -m "chore: update curation artifacts" }
    git push
}

if (-not $SkipVercel) {
    Write-Host "==> deploying web/ to Vercel" -ForegroundColor Cyan
    Push-Location web
    try { vercel --prod --yes } finally { Pop-Location }
}
