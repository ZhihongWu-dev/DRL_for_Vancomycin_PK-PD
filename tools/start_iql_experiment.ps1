param(
    [string]$config = "configs/iql_base.yaml",
    [string]$workdir = "algorithms/iql/runs/exp_manual",
    [int]$tbPort = 6006
)

# Run training in a new window and start TB
Start-Process -FilePath $env:PYTHON -ArgumentList "-m", "algorithms.iql.manage", "run-and-monitor", "--config", $config, "--workdir", $workdir, "--tb-port", $tbPort -NoNewWindow
Write-Host "Started training and monitor with config=$config workdir=$workdir tbPort=$tbPort"