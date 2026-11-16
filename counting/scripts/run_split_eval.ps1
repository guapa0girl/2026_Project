# 분할별 Geo / RF / RF+Geo / RF+Geo+Leg 평가 (test = 000) → compare_splits.py로 YOLO와 같이 비교.
# 포즈 모델 = training/models/results/ep200[_<split>]/m (training/scripts/run_split_study.ps1로 학습)
# RF = 그 분할의 train 사진(training/data/splits/<split>.json)으로만 학습
# 사용법 (counting/ 에서):  powershell -File scripts\run_split_eval.ps1 [-Python <python.exe>]
param([string]$Python = "python",
      [string[]]$Splits = @("random", "sorted", "k5_f0", "k5_f1", "k5_f2", "k5_f3", "k5_f4"))
$ErrorActionPreference = "Stop"
foreach ($s in $Splits) {
    $run = if ($s -eq "random") { "ep200" } else { "ep200_$s" }
    Write-Host "===== $s (pose $run) ====="
    & $Python scripts\evaluate_methods.py --cameras configs/eval_cameras.yaml `
        --stage1-xml data/labels/state_001.xml data/labels/state_002.xml `
        --stage2-xml data/labels/state_000.xml --stage2-images data/raw/000_img `
        --stage2-train-split "../training/data/splits/$s.json" `
        --pose-weights "../training/models/results/$run/m/weights/best.pt" `
        --out "outputs/eval/split_$s"
    if ($LASTEXITCODE -ne 0) { throw "eval $s failed" }
}
& $Python scripts\compare_splits.py
