# models/

가중치 파일은 git에 올리지 않습니다. [Releases](../../../releases)에서 받아 이 폴더에 둡니다.

| 파일 | 내용 | 쓰는 곳 |
|---|---|---|
| `pig_pose_ch010_m_best.pt` | YOLOv8m-pose, 287장(CH005·CH010·CH002 혼합), 100 epoch, imgsz 640, 2026-11-05. `training/models/results/m/weights/best.pt`와 같은 파일 | `configs/farm_ch010.yaml` |

- 이름의 `ch010`은 **이 가중치를 쓰는 곳(CH010 추론)**을 뜻합니다. 학습 데이터는 여러 채널이 섞여 있습니다.
- 새로 학습한 가중치로 바꿀 때는 config의 `model.weights`(와 `imgsz`)를 고치고, 바꾸기 전/후 결과를 섞어서 비교하지 마세요.
- keypoint 순서는 `training/README.md` 4장과 `src/feeding_monitor/pose_tracker.py`의 `KEYPOINT_NAMES`를 따릅니다.
  다른 CVAT 프로젝트에서 만든 가중치를 쓰기 전에는 실제 프레임에 keypoint 번호를 찍어 코(12)·목(11) 위치를 확인하세요.
