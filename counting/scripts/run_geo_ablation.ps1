# Geo 규칙 ablation - Stage 2 (YOLO keypoint). configsblationmeras_*.yaml 에 CH005, CH010, SAMPLE(000) 설정이 있음
# counting 폴더에서:  powershell -ExecutionPolicy Bypass -File scripts\run_geo_ablation.ps1 -Xml "<라벨 XML>" -Images "<이미지 폴더>"
param([string]$Xml, [string]$Images, [string]$Py = "..\.venv\Scripts\python")
$variants = "dir60", "dir90", "nodir", "dir60_posture", "nodir_posture"
foreach ($v in $variants) {
    Write-Host "`n===== $v =====" -ForegroundColor Cyan
    & $Py scripts\evaluate_methods.py --cameras "configs\ablation\cameras_$v.yaml" `
        --stage2-xml $Xml --stage2-images $Images `
        --out "outputs\eval\geo_ablation\$v"
}
Write-Host "`n결과: outputs\eval\geo_ablation\<variant>\summary.md" -ForegroundColor Green
