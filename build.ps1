# Build the camera-ready paper and the reviewer response, then check both
# against the iCONEECT 2026 requirements.
#
#   .\build.ps1              paper + response + verify
#   .\build.ps1 -Paper       paper only
#   .\build.ps1 -Full        regenerate tables and figures from the CSVs first
#
# pdflatex is run twice because cross-references and the bibliography need a
# second pass to settle. A single pass leaves "??" in the text.

param(
    [switch]$Paper,
    [switch]$Full
)

# Not "Stop": MiKTeX prints an update notice on stderr, and PowerShell 5.1
# turns any stderr from a native exe into a terminating NativeCommandError.
# Success is judged by $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$miktex = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
if (Test-Path "$miktex\pdflatex.exe") { $env:Path = "$miktex;$env:Path" }
$py = ".\.venv\Scripts\python.exe"

function Build($stem) {
    Write-Host "`n[latex] $stem" -ForegroundColor Cyan
    foreach ($pass in 1, 2) {
        # '>' redirects stdout only; stderr is left alone on purpose.
        pdflatex -interaction=nonstopmode -halt-on-error "$stem.tex" > "$stem.build.log"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAILED on pass ${pass}:" -ForegroundColor Red
            Select-String -Path "$stem.build.log" -Pattern '^!' |
                Select-Object -First 5 | ForEach-Object { "    " + $_.Line }
            exit 1
        }
    }
    $m = Select-String -Path "$stem.build.log" -Pattern 'Output written.*?\((\d+) pages'
    $pages = if ($m) { $m.Matches[0].Groups[1].Value } else { "?" }
    Write-Host "  ok - $pages pages" -ForegroundColor Green
}

if ($Full) {
    Write-Host "[data] regenerating tables and figures" -ForegroundColor Cyan
    & $py make_paper_tables.py
    & $py make_figures.py
}

Build "Iconeect2026_camera_ready"
if (-not $Paper) { Build "PID_Response" }

Write-Host "`n[check] compliance" -ForegroundColor Cyan
& $py verify_camera_ready.py

Write-Host "`n[copy] submission folder" -ForegroundColor Cyan
Copy-Item Iconeect2026_camera_ready.pdf camera_ready_submission\ -Force
Copy-Item PID_Response.pdf              camera_ready_submission\ -Force
Copy-Item Iconeect2026_camera_ready.tex camera_ready_submission\source\ -Force
Copy-Item PID_Response.tex              camera_ready_submission\source\ -Force
Write-Host "  camera_ready_submission\ updated" -ForegroundColor Green
