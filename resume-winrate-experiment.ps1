param([int]$Port=8766)
$ErrorActionPreference='Stop'
$experimentRoot=$PSScriptRoot
$runPath=(Get-Content -LiteralPath (Join-Path $experimentRoot 'latest_run.txt') -Raw).Trim()
$config=Get-Content -LiteralPath (Join-Path $runPath 'config.json') -Raw | ConvertFrom-Json
if($config.protocol -ne 'individual-pool-winrate-full-v1'){throw 'The current run uses a different training protocol.'}
$pidPath=Join-Path $experimentRoot 'training.pid'
if(Test-Path -LiteralPath $pidPath){
    $processId=[int](Get-Content -LiteralPath $pidPath)
    $info=Get-CimInstance Win32_Process -Filter "ProcessId = $processId"
    if($info -and ($info.CommandLine -like '*train.py*' -or $info.CommandLine -like '*algorithm_comparison.py*')){throw 'A trainer is still running.'}
}
$checkpoint=Join-Path $runPath 'latest.pt.gz'
if(-not(Test-Path -LiteralPath $checkpoint)){throw 'No complete checkpoint was found.'}
$stopPath=Join-Path $runPath 'STOP'
if(Test-Path -LiteralPath $stopPath){Remove-Item -LiteralPath $stopPath}
$trainArgs=@('-u','individual_train.py','--selection','win_rate','--reward',$config.reward.mode,
    '--compact-checkpoints','--publish','--out',"`"$runPath`"",'--resume',"`"$checkpoint`"")
foreach($key in @('models','games-per-opponent','win-threshold','initial-percent','algorithm','record-games',
    'record-every','patience','eval-games','eval-every','save-every','width','layers','heads','grad-chunk',
    'seed','updates','lr','ppo-epochs','ppo-minibatch','ppo-clip','gae-lambda','value-coef','target-kl','min-free-gib','device')){
    $trainArgs += @("--$key",[string]$config.($key.Replace('-','_')))
}
if($config.initial_checkpoint){$trainArgs+=@('--initial-checkpoint',"`"$($config.initial_checkpoint)`"")}
$pythonPath=(Get-Command python).Source
$resumeStamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$process=Start-Process -FilePath $pythonPath -ArgumentList $trainArgs -WorkingDirectory $experimentRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runPath "resume-$resumeStamp.stdout.log") -RedirectStandardError (Join-Path $runPath "resume-$resumeStamp.stderr.log") -PassThru
Set-Content -LiteralPath $pidPath -Value $process.Id
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
