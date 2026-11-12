# models/

가중치 파일은 git에 올리지 않음. 학습한 가중치를 이 폴더에 복사해서 사용.

| 파일 | 내용 | 쓰는 곳 |
|---|---|---|
| `pig_pose_best.pt` | YOLOv8-pose, 001(CH005)+002(CH010) 190장으로 학습한 n/s/m/l × epoch 100·200 중 test(000) 기준으로 고른 것. `training/models/results/<run>/<size>/weights/best.pt`를 복사 | `configs/farm_*.yaml`, `configs/ablation/*.yaml` |

- 바꿀 때는 config의 `model.weights`(와 `imgsz`)를 고치고, 바꾸기 전/후 결과를 섞어서 비교하지 말 것.
- keypoint 순서는 `training/README.md` 5장과 `src/feeding_monitor/pose_tracker.py`의 `KEYPOINT_NAMES`를 따름.
  다른 CVAT 프로젝트에서 만든 가중치를 쓰기 전에는 실제 프레임에 keypoint 번호를 찍어 코(12)·목(11) 위치를 확인할 것.
