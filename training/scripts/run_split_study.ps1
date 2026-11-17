# 분할 방식별 포즈 모델(medium, epoch 200, batch 8, imgsz 640, seed 42, best.pt = valid pose mAP50-95 최고) 학습.
# 끝나면 data/processed를 랜덤 분할로 되돌림.
# 사용법 (training/ 에서):
#   powershell -File scripts\run_split_study.ps1 [-Python <python.exe>]                        # 시간대 + 시간대 5-fold
#   powershell -File scripts\run_split_study.ps1 -Splits k5_f0,k5_f1,k5_f2,k5_f3,k5_f4,rk5_f0,rk5_f1,rk5_f2,rk5_f3,rk5_f4
param([string]$Python = "python", [string[]]$Splits = @("sorted", "k5_f0", "k5_f1", "k5_f2", "k5_f3", "k5_f4"))
$ErrorActionPreference = "Stop"
$Splits = $Splits | ForEach-Object { $_ -split "," } | Where-Object { $_ }   # -File 로 넘기면 "a,b"가 문자열 하나로 옴
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
