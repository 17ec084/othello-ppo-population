param([int]$Population=20,[int]$GamesPerOpponent=128,[int]$Updates=0,[int]$Seed=20261028,[int]$Port=8766,
    [ValidateSet('margin','signed_win')][string]$Reward='signed_win')
$ErrorActionPreference='Stop'
$experimentRoot=$PSScriptRoot
$pythonPath=(Get-Command python).Source
$checkIds=@()
foreach($name in @('training.pid','training-pids.json')){
    $path=Join-Path $experimentRoot $name
    if(Test-Path -LiteralPath $path){
        if($name -eq 'training.pid'){$checkIds += [int](Get-Content -LiteralPath $path)}
        else{$checkIds += (Get-Content -LiteralPath $path -Raw | ConvertFrom-Json).pids}
    }
}
foreach($processId in ($checkIds | Select-Object -Unique)){
    $info=Get-CimInstance Win32_Process -Filter "ProcessId = $processId"
    if($info -and ($info.CommandLine -like '*train.py*' -or $info.CommandLine -like '*algorithm_comparison.py*')){throw 'A trainer is already running.'}
}
$driveInfo=[IO.DriveInfo]::new([IO.Path]::GetPathRoot($experimentRoot))
if($driveInfo.AvailableFreeSpace -lt 4GB){throw 'At least 4 GiB of free disk space is required before starting.'}
$stamp=[TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTimeOffset]::UtcNow,'Tokyo Standard Time').ToString('yyyyMMdd-HHmmss')
$runPath=Join-Path $experimentRoot "runs\winrate-full-$stamp"
New-Item -ItemType Directory -Path $runPath | Out-Null
$snapshot=Join-Path $runPath 'source'
New-Item -ItemType Directory -Path $snapshot | Out-Null
Get-ChildItem -LiteralPath $experimentRoot -File | Where-Object {$_.Extension -in '.py','.js','.html','.ps1','.md' -and $_.Name -notlike 'test_*'} | Copy-Item -Destination $snapshot
$trainArgs=@('-u','individual_train.py','--selection','win_rate','--algorithm','ppo',
    '--models',"$Population",'--games-per-opponent',"$GamesPerOpponent",'--initial-percent','100',
    '--reward',$Reward,'--win-threshold','0.55','--grad-chunk','256','--ppo-minibatch','512',
    '--seed',"$Seed",'--updates',"$Updates",'--save-every','1','--min-free-gib','2','--compact-checkpoints','--publish','--out',"`"$runPath`"")
$process=Start-Process -FilePath $pythonPath -ArgumentList $trainArgs -WorkingDirectory $experimentRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runPath 'stdout.log') -RedirectStandardError (Join-Path $runPath 'stderr.log') -PassThru
Set-Content -LiteralPath (Join-Path $experimentRoot 'training.pid') -Value $process.Id
@{pids=@($process.Id)} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $experimentRoot 'training-pids.json')
$monitorPidPath=Join-Path $experimentRoot 'monitor.pid'
$monitorRunning=$false
if(Test-Path -LiteralPath $monitorPidPath){
    $monitorId=[int](Get-Content -LiteralPath $monitorPidPath)
    $info=Get-CimInstance Win32_Process -Filter "ProcessId = $monitorId"
    $monitorRunning=$info -and $info.CommandLine -like '*monitor.py*'
}
if(-not $monitorRunning){
    $monitor=Start-Process -FilePath $pythonPath -ArgumentList @('-u','monitor.py','--port',"$Port") -WorkingDirectory $experimentRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $experimentRoot 'monitor.stdout.log') -RedirectStandardError (Join-Path $experimentRoot 'monitor.stderr.log') -PassThru
    Set-Content -LiteralPath $monitorPidPath -Value $monitor.Id
}
@{RunDirectory=$runPath;TrainingProcess=$process.Id;Monitor="http://127.0.0.1:$Port/"} | ConvertTo-Json
