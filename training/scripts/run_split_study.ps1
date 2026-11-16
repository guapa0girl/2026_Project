# 분할 방식별 포즈 모델(medium, epoch 200) 학습: 시간대(sorted) + 시간대 5-fold(k5_f0~f4).
# 랜덤 분할은 기존 ep200 run을 그대로 씀. 끝나면 data/processed를 랜덤 분할로 되돌림.
# 사용법 (training/ 에서):  powershell -File scripts\run_split_study.ps1 [-Python <python.exe>]
param([string]$Python = "python", [string[]]$Splits = @("sorted", "k5_f0", "k5_f1", "k5_f2", "k5_f3", "k5_f4"))
$ErrorActionPreference = "Stop"
& $Python scripts\make_study_splits.py
foreach ($s in $Splits) {
    Write-Host "===== split $s ====="
    & $Python scripts\split_dataset.py --split-file "data\splits\$s.json"
    if ($LASTEXITCODE -ne 0) { throw "split $s failed" }
    $env:POSE_RUN = "ep200_$s"
    & $Python scripts\train.py --sizes m --epochs 200 --patience 0
    if ($LASTEXITCODE -ne 0) { throw "train $s failed" }
}
$env:POSE_RUN = ""
& $Python scripts\split_dataset.py --split-file data\splits\random.json
Write-Host "split study done"
