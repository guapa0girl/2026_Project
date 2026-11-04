"""configs/pig_pose.yaml 에 정의된 18개 키포인트 이름/스켈레톤/flip_idx 를
학습 스크립트가 아닌 다른 스크립트(평가, 시각화, 실시간 추론)에서도
동일하게 재사용하기 위한 헬퍼."""
import yaml

from .paths import DATA_YAML, PROCESSED_DIR, RESOLVED_DATA_YAML


def load_pose_spec(data_yaml=DATA_YAML):
    with open(data_yaml, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    return {
        "num_keypoints": spec["kpt_shape"][0],
        "keypoint_names": spec["keypoint_names"],
        "skeleton": [tuple(edge) for edge in spec["skeleton"]],
        "flip_idx": spec["flip_idx"],
    }


def resolve_data_yaml():
    """configs/pig_pose.yaml을 읽어 'path'를 data/processed의 절대경로로
    바꾼 사본을 만들어 반환한다. Ultralytics의 model.train()/model.val()에는
    항상 이 함수의 반환값을 넘겨야 한다 — 'path'가 상대경로인 채로 넘기면
    실행 시점 CWD에 따라 다르게(심하면 이 프로젝트가 아닌 전혀 다른 폴더로)
    해석될 수 있다."""
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    spec["path"] = str(PROCESSED_DIR.resolve())

    RESOLVED_DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(RESOLVED_DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, allow_unicode=True, sort_keys=False)
    return RESOLVED_DATA_YAML
