"""학습 산출물(Joblib·JSON·CSV)과 YOLO Pose 가중치를 불러오는 모듈.

서버가 시작될 때 load_bundle()을 한 번만 호출하고, 이후 요청에서는
같은 ModelBundle 객체를 재사용합니다. (요청마다 모델을 다시 읽지 않음)

- action_model.joblib : robust2 창(2초) 모델. 입력 175차원 = 35 Frame Feature × 5 통계
- clip_model.joblib   : 영상(트랙) 전체 요약 모델. 입력 503차원
- model_info.json     : 학습 설정 (window_sec, fps, stride_sec, classes …)
- action_classes.csv  : A001~A017 ↔ 영문 label ↔ 한글 설명
- feature_columns.json: 두 모델이 기대하는 Feature 이름과 순서 (scripts/check_model.py가 생성)

⚠ joblib.load()는 직접 학습한 신뢰 가능한 파일에만 사용합니다.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib

AI_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("MODEL_DIR", AI_ROOT / "models"))
PROJECT_CONFIG = Path(os.getenv("PROJECT_CONFIG", AI_ROOT / "project.json"))

# ----------------------------------------------------
# Feature 이름 (src/motion_capture_features.py의 계산 순서와 동일)
# ----------------------------------------------------
ROBUST_NAMES = [
    "elbow_angle_lo", "knee_angle_lo", "armpit_angle_lo", "hip_angle_lo",
    "elbow_angle_hi", "knee_angle_hi", "armpit_angle_hi", "hip_angle_hi",
    "trunk_tilt",
    "wrist_height_lo", "wrist_height_hi", "ankle_height_lo", "ankle_height_hi",
    "nose_height", "hip_y",
]
EXTENDED_NAMES = [
    "arm_reach_lo", "arm_reach_hi", "leg_reach_lo", "leg_reach_hi",
    "upperarm_len_lo", "upperarm_len_hi", "thigh_len_lo", "thigh_len_hi",
    "wrist_gap", "ankle_gap", "ankle_dy", "wrist_dy", "nose_hip_dist",
    "height_width_ratio",
    "wrist_speed_lo", "wrist_speed_hi", "ankle_speed_lo", "ankle_speed_hi",
    "nose_speed", "hip_speed",
]
FRAME_FEATURES = ROBUST_NAMES + EXTENDED_NAMES  # 35개
VERTICAL_NAMES = ["vy_hip", "vy_nose", "vy_shoulder"]
WINDOW_STATS = ["mean", "std", "min", "max", "absvel"]


def window_feature_columns() -> list[str]:
    return [f"{stat}__{name}" for stat in WINDOW_STATS for name in FRAME_FEATURES]


def clip_feature_columns(parts: int = 10) -> list[str]:
    base = FRAME_FEATURES + VERTICAL_NAMES
    columns = [f"part{k:02d}__{name}" for k in range(parts) for name in base]
    for stat in ("std", "min", "max"):
        columns += [f"{stat}__{name}" for name in base]
    for stat in ("smooth_max", "smooth_min", "smooth_argmax"):
        columns += [f"{stat}__{name}" for name in VERTICAL_NAMES]
    return columns


@dataclass
class ModelBundle:
    window_model: object
    clip_model: object
    info: dict
    project: dict
    actions: dict[str, dict] = field(default_factory=dict)
    pose_model_path: Path = MODEL_DIR / "yolo11n-pose.pt"
    tracker_path: Path = AI_ROOT / "bytetrack_stable.yaml"

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.window_model.classes_]

    def summary(self) -> dict:
        return {
            "model_type": "RandomForest hybrid (window + clip)",
            "condition": self.info.get("condition"),
            "classes": [
                {"action_id": c, **self.actions.get(c, {})} for c in self.classes
            ],
            "window_features": int(self.window_model.n_features_in_),
            "clip_features": int(self.clip_model.n_features_in_),
            "window_sec": float(self.info["window_sec"]),
            "stride_sec": float(self.info["stride_sec"]),
            "train_fps": float(self.info["fps"]),
            "pose_model": self.pose_model_path.name,
        }


def load_actions(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig") as file:
        return {
            row["action_id"]: {
                "action_label": row["action_label"],
                "description": row["description"],
            }
            for row in csv.DictReader(file)
        }


def load_bundle(model_dir: Path = MODEL_DIR) -> ModelBundle:
    """모델 파일을 읽고 학습 당시 규격과 맞는지 검사합니다. 어긋나면 RuntimeError."""
    info = json.loads((model_dir / "model_info.json").read_text(encoding="utf-8"))
    project = json.loads(PROJECT_CONFIG.read_text(encoding="utf-8"))
    actions = load_actions(model_dir / "action_classes.csv")

    if info.get("feature") != "hybrid" or info.get("window_feature") != "robust2":
        raise RuntimeError("hybrid(robust2 창 + clip) 모델만 지원합니다.")

    window_model = joblib.load(model_dir / "action_model.joblib")
    clip_model = joblib.load(model_dir / info["clip_model"])

    expected_window = len(window_feature_columns())
    expected_clip = len(clip_feature_columns())
    if int(window_model.n_features_in_) != expected_window:
        raise RuntimeError(
            f"창 모델 입력 {window_model.n_features_in_} ≠ 기대 {expected_window}"
        )
    if int(clip_model.n_features_in_) != expected_clip:
        raise RuntimeError(
            f"Clip 모델 입력 {clip_model.n_features_in_} ≠ 기대 {expected_clip}"
        )
    if list(window_model.classes_) != list(clip_model.classes_):
        raise RuntimeError("두 모델의 classes_ 순서가 다릅니다.")
    if [str(c) for c in window_model.classes_] != list(info["classes"]):
        raise RuntimeError("model_info.json의 classes와 모델 classes_가 다릅니다.")
    missing = set(info["classes"]) - set(actions)
    if missing:
        raise RuntimeError(f"action_classes.csv에 없는 클래스: {sorted(missing)}")

    pose_path = model_dir / project["pose_model"]
    if not pose_path.exists():
        raise RuntimeError(f"Pose 가중치가 없습니다: {pose_path}")

    return ModelBundle(
        window_model=window_model,
        clip_model=clip_model,
        info=info,
        project=project,
        actions=actions,
        pose_model_path=pose_path,
        tracker_path=AI_ROOT / project.get("tracker", "bytetrack_stable.yaml"),
    )
