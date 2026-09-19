param(
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run: python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt"
}

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    throw "cloudflared is not installed. Run: winget install Cloudflare.cloudflared"
}

$env:ALLOW_RESTART = "0"
$server = Start-Process `
    -FilePath $python `
    -ArgumentList "-u", "live_server.py", "--host", "127.0.0.1", "--port", $Port, "--no-open" `
    -WorkingDirectory $root `
    -PassThru

try {
    Write-Host "Starting public tunnel. Share the https://*.trycloudflare.com URL printed below."
    Write-Host "Press Ctrl+C to stop the tunnel and training server."
    cloudflared tunnel --url "http://127.0.0.1:$Port"
}
finally {
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
}
