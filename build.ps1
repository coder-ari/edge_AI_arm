# Auto-generated build script for Windows (PowerShell + gcc)
# Pure-C per-channel int8 inference — no CMSIS source files needed.
# Only CMSIS\Include\ is required for the type definitions.
#
# Usage:  .\build.ps1
# Run:    .\my_model_test.exe

# Collect include dirs from CMSIS recursively (for arm_nn_types.h etc.)
$inc_dirs = @("-I.", "-ICMSIS\Include")
if (Test-Path "CMSIS") {
    $sub = Get-ChildItem -Path CMSIS -Recurse -Directory |
           ForEach-Object { "-I$($_.FullName)" }
    $inc_dirs = $inc_dirs + $sub | Select-Object -Unique
}

$srcs   = @("main.c", "generated\my_model_run.c")
$flags  = $inc_dirs + @("-O2", "-lm")
$output = "my_model_test.exe"

Write-Host "[build] Compiling -> $output"
gcc @flags -o $output @srcs

if ($LASTEXITCODE -eq 0) {
    Write-Host "[build] OK  ->  .\$output"
} else {
    Write-Host "[build] FAILED (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}
