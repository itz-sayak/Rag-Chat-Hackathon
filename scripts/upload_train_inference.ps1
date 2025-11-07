<#
PowerShell helper to upload `train_validation_inference/A` to remote repository.

USAGE (run as Administrator from the repo root):
  powershell -ExecutionPolicy Bypass -File .\scripts\upload_train_inference.ps1

What it does:
- Attempts to remove stale .git/index.lock
- Takes ownership of .git and grants full control to the current user (recursive)
- Adds .gitignore and `train_validation_inference/A` to the index, commits, and pushes to origin/main

IMPORTANT: Run this script only if you trust it and on your local machine. It will modify file ACLs for the local repository folder.
#>

function Ensure-RunningAsAdmin {
    $current = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $current.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Host "This script must be run as Administrator. Right-click PowerShell and choose 'Run as Administrator'." -ForegroundColor Red
        exit 1
    }
}

try {
    Ensure-RunningAsAdmin
    $repoRoot = Resolve-Path -Path "."
    Write-Host "Repository root: $repoRoot"

    # Remove stale index.lock if present
    $lockPath = Join-Path $repoRoot ".git\index.lock"
    if (Test-Path $lockPath) {
        Write-Host "Removing stale index.lock..." -ForegroundColor Yellow
        Remove-Item -Path $lockPath -Force -ErrorAction SilentlyContinue
    }

    # Take ownership and grant full control to current user for .git
    Write-Host "Taking ownership of .git and granting current user full control (this requires Admin)..." -ForegroundColor Yellow
    & takeown /f "$repoRoot\.git" /r /d Y
    & icacls "$repoRoot\.git" /grant "$env:USERNAME:(F)" /T

    # Confirm .git is writable by trying to create a small temp file
    $testFile = Join-Path $repoRoot ".git\.upload_test"
    Set-Content -Path $testFile -Value "ok" -Force
    Remove-Item -Path $testFile -Force

    # Stage files
    Write-Host "Staging .gitignore and train_validation_inference/A ..." -ForegroundColor Green
    git add .gitignore
    git add train_validation_inference/A

    # Commit
    $msg = "Add training/inference scripts and small outputs (exclude checkpoints/datasets)"
    git commit -m "$msg"

    # Push
    Write-Host "Pushing to origin main..." -ForegroundColor Green
    git push origin main

    Write-Host "Done. If any step failed above, review the output and retry." -ForegroundColor Cyan
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    exit 2
}
