$ErrorActionPreference = 'Stop'
$runPath = (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'latest_run.txt') -Raw).Trim()
$runPaths = @($runPath)
$comparisonPointer = Join-Path $PSScriptRoot 'latest_comparison.txt'
if (Test-Path -LiteralPath $comparisonPointer) {
    $comparisonRoot = (Get-Content -LiteralPath $comparisonPointer -Raw).Trim()
    $manifestPath=Join-Path $comparisonRoot 'comparison.json'
    if(Test-Path -LiteralPath $manifestPath){
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $candidatePaths = @($manifest.runs.PSObject.Properties | ForEach-Object { $_.Value.path })
        if ($candidatePaths -contains $runPath) { $runPaths = $candidatePaths }
    }
}
foreach ($path in $runPaths) { New-Item -ItemType File -Path (Join-Path $path 'STOP') -Force | Out-Null }
Write-Output "Stop requested for $($runPaths.Count) run(s). Each trainer will save after its current update."
