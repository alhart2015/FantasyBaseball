$ErrorActionPreference = "Stop"

Write-Host "=== 1/4 Unit tests ===" -ForegroundColor Cyan
python -m pytest tests/ -v
if ($LASTEXITCODE -ne 0) { throw "Unit tests failed" }
Write-Host ""

Write-Host "=== 2/4 Sync Redis ===" -ForegroundColor Cyan
python scripts/sync_redis.py
if ($LASTEXITCODE -ne 0) { throw "Sync Redis failed" }
Write-Host ""

Write-Host "=== 3/4 Smoke test ===" -ForegroundColor Cyan
python scripts/smoke_test.py
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed" }
Write-Host ""

Write-Host "=== 4/4 Run lineup ===" -ForegroundColor Cyan
python scripts/run_lineup.py
if ($LASTEXITCODE -ne 0) { throw "Run lineup failed" }
Write-Host ""

Write-Host "=== All checks passed ===" -ForegroundColor Green
Read-Host "Press Enter to close"
