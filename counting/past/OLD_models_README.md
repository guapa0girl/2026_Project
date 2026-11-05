# models/

Deployed YOLO-Pose checkpoints (gitignored -- too large for version control).
There are four checkpoint lineages here; `configs/pen01.yaml` uses the first:

- **`pig_pose_m_best.pt`** -- generic default. YOLOv8m-pose, 200 epochs,
  imgsz **640** (must match at inference -- other checkpoints here were
  trained at 960), copied from `../../results/m/weights/best.pt`. That
  `results/` folder holds a full nano/small/medium/large size sweep from the
  separate `auto_labeling_model` project (`../../results/{n,s,m,l}/`, each
  with `args.yaml` + `results.csv`); medium had the best pose mAP50-95
  (0.508) of the four at final epoch, edging out large (0.492) and small
  (0.459), so it's the one deployed here. To switch to another size, copy
  e.g. `results/l/weights/best.pt` in and update `model.weights`/`imgsz`.

- **`pig_pose_v2_m_best.pt`** -- used by the real farm (CH005) configs
  (`configs/farm_ch005_v2m_*.yaml`, 2026-08-25). Copied from
  `../../models_v2/results/m/weights/best.pt`, a separate n/s/m/l sweep
  (same `auto_labeling_model` project, `configs/_pig_pose.resolved.yaml` --
  same 18-keypoint node-id order as above, so `pose_tracker.KEYPOINT_NAMES`
  applies unchanged). Medium again had the best pose mAP50-95 of this sweep
  (0.514 vs n=0.408/s=0.469/l=0.495), imgsz 640.

- **`trained/train1_best.pt` .. `train5_best.pt`** -- an earlier lineage.
  Each round finetunes from the previous, 200 epochs/round, imgsz 960,
  starting from `pretrained/yolov8n-pose.pt`. `train5_best.pt` (2026-08-14)
  is the latest of these. `trained/best.pt` (2026-08-20, from a separate
  "pig_pose_combined" run, larger architecture, ~21M params) is a distinct,
  later checkpoint that happens to share the same filename convention.

- **`pig_pose_v3_best.pt`** -- copy of `../training/pig_pose/bootstrap_v3/weights/best.pt`,
  the CVAT auto-label/correct/retrain loop in `../training/README.md`
  (`bootstrap_v1` -> `v2` -> `v3`, 2026-08-04).

All four lineages share the same 18-keypoint pig skeleton (`task=pose`,
`kpt_shape=[18,3]`, `names={0:'pig'}`) -- **but see the keypoint-order
warning below before trusting a new checkpoint's nose/neck output.**

## Keypoint order -- read before adding a new checkpoint

The 18 keypoints are `tail_tip, tail, l_hip, r_hip, l_foot, r_foot, r_flank,
l_flank, l_shoulder, r_shoulder, l_hand, neck, nose, l_ear, l_ear_tip, r_ear,
r_ear_tip, r_hand` (0-indexed in that order -- `nose_index=12`, `neck_index=11`
in every `configs/*.yaml`). This is the CVAT skeleton label's SVG
**`data-node-id` ascending order (creation order)**, confirmed against
`../../auto_labeling_model/configs/pig_pose.yaml` (a separate, more mature
pipeline that independently hit and documented this exact issue).

It is **not** the label config JSON's `"sublabels"` array display order --
that array can be freely reordered in the CVAT UI without changing the
already-exported point order, and assuming otherwise is a real bug this
project shipped once already: it silently used `l_ear` as "nose" throughout
the feeding classifier and drew garbled skeleton connectivity, because
sublabels-order puts "nose" last (index 17) instead of right after "neck"
(index 12). If you regenerate the CVAT skeleton or add a new checkpoint from
a differently-configured label project, verify the order against real
labeled data before trusting it -- render a ground-truth frame's keypoints
with their index numbers on the actual image and check nose/neck land where
expected, rather than trusting either JSON array's order at face value.

To switch which checkpoint a pen uses, edit that config's `model.weights`
(and `imgsz` if it differs). After a new `../training/train_next.py` round:

```bash
cp training/pig_pose/bootstrap_v4/weights/best.pt models/pig_pose_v4_best.pt
```

## `posture_random_forest.joblib`

Not a pose checkpoint -- a pretrained RandomForest **feeding classifier**
(comparison-only, see the README's "두 번째 판정 방식" section), copied from
`auto_labeling_model/models/posture/random_forest.joblib`. Trained on ~940
hand-labeled frames from the same CH005 farm camera. Its own accuracy is
only validated against the rule labels it was bootstrapped from (100% by
construction -- see that project's `reports/posture/classifier_report.json`)
-- not against real human judgment, so treat it the same as the geometric
classifier for scripts/validate_counts.py purposes: unproven until scored
against real `human_count` labels.
