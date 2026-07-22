# run_analyses.ps1
# Runs hap40_interactome.jl and hap40_differential_interactome.jl sequentially,
# logs all output, and optionally shuts down the PC after completion.
#
# Usage:  powershell -ExecutionPolicy Bypass -File examples\run_analyses.ps1 [-Shutdown]
# From:   The BayesInteractomics repository root

param(
    [switch]$Shutdown
)

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$repoRoot = Split-Path -Parent $PSScriptRoot
$examplesDir = $PSScriptRoot
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$logFile = Join-Path $repoRoot "analysis_log_$timestamp.txt"
$threads = 30

Write-Host "Log file: $logFile"
Write-Host "Threads:  $threads"
Write-Host ""

# Helper: run a Julia script, log output, continue on error
function Invoke-JuliaScript {
    param([string]$ScriptPath, [string]$Label)

    $separator = "=" * 70
    $startTime = Get-Date

    Add-Content -Path $logFile -Value ""
    Add-Content -Path $logFile -Value $separator
    Add-Content -Path $logFile -Value "$Label"
    Add-Content -Path $logFile -Value "Script: $ScriptPath"
    Add-Content -Path $logFile -Value "Started: $startTime"
    Add-Content -Path $logFile -Value $separator

    Write-Host "$separator"
    Write-Host "Starting: $Label"
    Write-Host "  Script: $ScriptPath"
    Write-Host "  Time:   $startTime"
    Write-Host "$separator"

    try {
        # ForEach-Object {"$_"} converts stderr ErrorRecords to plain strings
        # before Tee-Object sees them, preventing wide formatted output.
        & julia --threads=$threads --project="$examplesDir" "$ScriptPath" 2>&1 |
            ForEach-Object { "$_" } |
            Tee-Object -FilePath $logFile -Append
        $exitCode = $LASTEXITCODE
    }
    catch {
        $exitCode = 1
        $errMsg = "EXCEPTION: $_"
        Add-Content -Path $logFile -Value $errMsg
        Write-Host $errMsg -ForegroundColor Red
    }

    $endTime = Get-Date
    $duration = $endTime - $startTime
    $statusLine = if ($exitCode -eq 0) { "COMPLETED SUCCESSFULLY" } else { "FAILED (exit code: $exitCode)" }

    $summary = @"

$separator
$Label -- $statusLine
Duration: $($duration.ToString('hh\:mm\:ss'))
Ended:    $endTime
$separator

"@
    Add-Content -Path $logFile -Value $summary
    Write-Host $summary

    return $exitCode
}

# --- Header ---
$header = @"
BayesInteractomics Analysis Run
================================
Date:    $(Get-Date)
Threads: $threads
Repo:    $repoRoot

Scripts:
  1) hap40_interactome.jl
  2) hap40_differential_interactome.jl
"@
Set-Content -Path $logFile -Value $header
Write-Host $header

# --- Run scripts ---
$exit1 = Invoke-JuliaScript `
    -ScriptPath (Join-Path $examplesDir "hap40_interactome.jl") `
    -Label "[1/2] HAP40 Interactome Analysis + Docking Request Generation"

$exit2 = Invoke-JuliaScript `
    -ScriptPath (Join-Path $examplesDir "hap40_differential_interactome.jl") `
    -Label "[2/2] HAP40 Differential Interactome Analysis"

# --- Summary ---
$finalSummary = @"

========================================
FINAL SUMMARY
========================================
hap40_interactome.jl:              $(if ($exit1 -eq 0) { 'OK' } else { "FAILED ($exit1)" })
hap40_differential_interactome.jl: $(if ($exit2 -eq 0) { 'OK' } else { "FAILED ($exit2)" })
Log: $logFile
========================================

$(if ($Shutdown) { 'Backup & shutdown pending...' } else { 'Done. (Pass -Shutdown to backup and shut down.)' })
"@
Add-Content -Path $logFile -Value $finalSummary
Write-Host $finalSummary

# --- Backup & Shutdown ---
if ($Shutdown) {
    $backupDest = "E:\Bayesinteractomics"
    $sources = @(
        "C:\Users\Manuel\Documents\GitHub\BayesInteractomics",
        "C:\Users\Manuel\Desktop\HAP40_interactome_enrichment"
    )

    Write-Host ""
    Write-Host "Copying folders to $backupDest ..."
    $backupOk = $true
    foreach ($src in $sources) {
        $folderName = Split-Path -Leaf $src
        $dest = Join-Path $backupDest $folderName
        Write-Host "  $src -> $dest"
        try {
            Copy-Item -Path $src -Destination $dest -Recurse -Force -ErrorAction Stop
            if (-not (Test-Path $dest)) {
                throw "Destination folder $dest does not exist after copy."
            }
            Write-Host "    OK" -ForegroundColor Green
        }
        catch {
            Write-Host "    FAILED: $_" -ForegroundColor Red
            Add-Content -Path $logFile -Value "BACKUP FAILED: $src -> $dest : $_"
            $backupOk = $false
        }
    }

    if ($backupOk) {
        Write-Host ""
        Write-Host "Backup complete." -ForegroundColor Green
        $shutdownMsg = "Shutting down in 5 minutes... (Ctrl+C to cancel)"
        Add-Content -Path $logFile -Value "Backup successful. $shutdownMsg"
        Write-Host $shutdownMsg

        Start-Sleep -Seconds 300
        Stop-Computer -Force
    } else {
        $abortMsg = "Backup failed - shutdown ABORTED. PC stays on."
        Add-Content -Path $logFile -Value $abortMsg
        Write-Host ""
        Write-Host $abortMsg -ForegroundColor Red
    }
} else {
    Write-Host ""
    Write-Host "All done. (Pass -Shutdown to backup and shut down the PC.)"
}
