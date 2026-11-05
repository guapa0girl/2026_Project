# Geo 규칙 ablation (E3) - Stage 2 (평가 50장, YOLO keypoint)
# counting 폴더에서:  powershell -ExecutionPolicy Bypass -File scripts\run_geo_ablation.ps1 -Xml "data\labels\ch010_eval50.xml" -Images "D:\eval50"
param([string]$Xml, [string]$Images, [string]$Py = "..\.venv\Scripts\python")
$variants = "dir60", "dir90", "nodir", "dir60_posture", "nodir_posture"
foreach ($v in $variants) {
    Write-Host "`n===== $v =====" -ForegroundColor Cyan
    & $Py scripts\evaluate_methods.py --cameras "configs\ablation\cameras_$v.yaml" `
        --stage2-xml $Xml --stage2-images $Images --stage2-only-list ..\training\eval_frames.txt `
        --out "outputs\eval\geo_ablation\$v"
}
Write-Host "`n결과: outputs\eval\geo_ablation\<variant>\summary.md" -ForegroundColor Green
