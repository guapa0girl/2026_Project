# 분할별 Geo / RF / RF+Geo / RF+Geo+Leg 평가 → compare_splits.py / compare_folds.py로 YOLO와 같이 비교.
# 포즈 모델 = training/models/results/ep200[_<split>]/m (training/scripts/run_split_study.ps1로 학습)
# RF = 그 분할의 train 사진(training/data/splits/<split>.json)으로만 학습
# test = 000 → outputs/eval/split_<split>/
# -Valid: 그 분할의 valid 사진(001+002)도 평가 → outputs/eval/split_<split>_val/
#         (사진은 training에서 make_study_splits.py --export-images ..\counting\data\raw\train_img 로 꺼냄)
# 사용법 (counting/ 에서):
#   powershell -File scripts\run_split_eval.ps1 [-Python <python.exe>]                 # 랜덤·시간대 단일 분할 + 시간대 5-fold (test)
#   powershell -File scripts\run_split_eval.ps1 -Valid -Splits k5_f0,k5_f1,k5_f2,k5_f3,k5_f4,rk5_f0,rk5_f1,rk5_f2,rk5_f3,rk5_f4
param([string]$Python = "python",
      [string[]]$Splits = @("random", "sorted", "k5_f0", "k5_f1", "k5_f2", "k5_f3", "k5_f4"),
      [switch]$Valid)
$ErrorActionPreference = "Stop"
$Splits = $Splits | ForEach-Object { $_ -split "," } | Where-Object { $_ }   # -File 로 넘기면 "a,b"가 문자열 하나로 옴
foreach ($s in $Splits) {
    $run = if ($s -eq "random") { "ep200" } else { "ep200_$s" }
    $common = @("--cameras", "configs/eval_cameras.yaml",
                "--stage1-xml", "data/labels/state_001.xml", "data/labels/state_002.xml",
                "--stage2-train-split", "../training/data/splits/$s.json",
                "--pose-weights", "../training/models/results/$run/m/weights/best.pt")
    Write-Host "===== $s (pose $run) test ====="
    & $Python scripts\evaluate_methods.py @common `
        --stage2-xml data/labels/state_000.xml --stage2-images data/raw/000_img --out "outputs/eval/split_$s"
    if ($LASTEXITCODE -ne 0) { throw "eval $s failed" }
    if ($Valid) {
        Write-Host "===== $s (pose $run) val ====="
        & $Python scripts\evaluate_methods.py @common --stage2-eval-part valid `
            --stage2-xml data/labels/state_001.xml data/labels/state_002.xml --stage2-images data/raw/train_img `
            --out "outputs/eval/split_${s}_val"
        if ($LASTEXITCODE -ne 0) { throw "eval $s val failed" }
    }
}
if ($Splits -contains "random" -or $Splits -contains "sorted") { & $Python scripts\compare_splits.py }
if ($Splits | Where-Object { $_ -like "*k5_f*" }) { & $Python scripts\compare_folds.py }
