# ========================================================
# Bureaucracy AI — All-in-One Live Deployment Launcher
# ========================================================
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   Starting Bureaucracy AI (Live System)  " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$ROOT = "C:\projects\IND_Bureaucracy-ai"
Set-Location $ROOT

# 1. Check Ollama
Write-Host "`n[1/3] Checking Ollama..." -ForegroundColor Yellow
$ollamaProc = Get-Process ollama -ErrorAction SilentlyContinue
if (-not $ollamaProc) {
    Write-Host "Starting Ollama in background..." -ForegroundColor Gray
    Start-Process -FilePath "C:\Users\elige\AppData\Local\Programs\Ollama\ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
} else {
    Write-Host "Ollama is already running." -ForegroundColor Green
}

# 2. Start FastAPI Backend (Port 8000)
Write-Host "`n[2/3] Launching FastAPI Backend on port 8000..." -ForegroundColor Yellow
$backendCmd = @"
`$env:PYTHONPATH='$ROOT'
`$env:HF_HUB_OFFLINE='1'
`$env:TRANSFORMERS_OFFLINE='1'
cd '$ROOT'
& '$ROOT\.venv\Scripts\python.exe' -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
"@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd -WindowStyle Minimized
Write-Host "Backend started on http://127.0.0.1:8000" -ForegroundColor Green
Start-Sleep -Seconds 3

# 3. Start Streamlit Frontend (Port 8501)
Write-Host "`n[3/3] Launching Streamlit Frontend on port 8501..." -ForegroundColor Yellow
$frontendCmd = @"
`$env:PYTHONPATH='$ROOT'
cd '$ROOT'
& '$ROOT\.venv\Scripts\python.exe' -m streamlit run frontend\app.py --server.port 8501 --server.headless false
"@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd -WindowStyle Normal
Write-Host "Frontend running at http://localhost:8501" -ForegroundColor Green
Start-Sleep -Seconds 2

# 4. Optional: Start Cloudflare Tunnel for Live Public HTTPS Link
Write-Host "`nDo you also want to generate a public live HTTPS link for external evaluators? (y/n): " -ForegroundColor Cyan -NoNewline
$resp = Read-Host
if ($resp -match "^[yY]") {
    Write-Host "Creating public HTTPS tunnel..." -ForegroundColor Yellow
    $tunnelCmd = @"
cd '$ROOT'
& '$ROOT\cloudflared.exe' tunnel --url http://127.0.0.1:8501
"@
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $tunnelCmd -WindowStyle Normal
}

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "All systems operational!" -ForegroundColor Green
Write-Host "Local URL: http://localhost:8501" -ForegroundColor White
Write-Host "Swagger Docs: http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
