from __future__ import annotations

# ====================================================
# Motion Capture 동작 인식 Feature (학습·실시간 공용)
# ====================================================
# 48번 학습 스크립트와 49번 실시간 스크립트가 같은 함수를 써야
# 학습 때와 실시간의 Feature 계산이 정확히 같아집니다.
#
# 입력 DataFrame: <관절>_x_px, <관절>_y_px 컬럼 (COCO 17관절, 픽셀 좌표)

import warnings

import numpy as np
import pandas as pd

from src.pose_utils import COCO_KEYPOINT_NAMES

JOINT = {name: index for index, name in enumerate(COCO_KEYPOINT_NAMES)}


# ----------------------------------------------------
# robust Feature: 방향에 덜 민감한 Frame Feature 16개
# ----------------------------------------------------
def _angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def robust_frame_features(df: pd.DataFrame) -> np.ndarray:
    points = np.stack(
        [
            df[[f"{n}_x_px", f"{n}_y_px"]].to_numpy(dtype=float)
            for n in COCO_KEYPOINT_NAMES
        ],
        axis=1,
    )

    def g(name):
        return points[:, JOINT[name]]

    shoulder = (g("left_shoulder") + g("right_shoulder")) / 2
    hip = (g("left_hip") + g("right_hip")) / 2
    torso = np.nanmedian(np.linalg.norm(shoulder - hip, axis=1))

    columns = []
    for side in ("left", "right"):
        columns += [
            _angle(g(f"{side}_shoulder"), g(f"{side}_elbow"), g(f"{side}_wrist")),
            _angle(g(f"{side}_hip"), g(f"{side}_knee"), g(f"{side}_ankle")),
            _angle(g(f"{side}_elbow"), g(f"{side}_shoulder"), g(f"{side}_hip")),
            _angle(g(f"{side}_shoulder"), g(f"{side}_hip"), g(f"{side}_knee")),
        ]
    trunk = shoulder - hip
    columns.append(np.degrees(np.arctan2(np.abs(trunk[:, 0]), -trunk[:, 1])))
    for name in ("left_wrist", "right_wrist", "left_ankle", "right_ankle", "nose"):
        columns.append((g(name)[:, 1] - hip[:, 1]) / torso)
    columns.append(hip[:, 1] / torso)

    features = np.stack(columns, axis=1)
    # 좌우 쌍을 (작은 값, 큰 값)으로 정렬 → 좌우 반전에 무관
    for a, b in [(0, 4), (1, 5), (2, 6), (3, 7), (9, 10), (11, 12)]:
        low = np.minimum(features[:, a], features[:, b])
        high = np.maximum(features[:, a], features[:, b])
        features[:, a], features[:, b] = low, high
    return features


# ----------------------------------------------------
# robust2 추가 Frame Feature 20개 (robust 16개 뒤에 붙여 36개로 사용)
#   팔·다리 뻗은 거리, 손·발 사이 거리, 몸 세로/가로 비율,
#   손목·발목·코 속도(엉덩이 기준), 엉덩이 이동 속도. 모두 몸통 길이로 정규화.
#   → 던지기·샷건·앞차기처럼 "얼마나 빨리, 멀리 뻗는지"가 핵심인 동작 구분용
# ----------------------------------------------------
def extended_frame_features(df: pd.DataFrame) -> np.ndarray:
    points = np.stack(
        [
            df[[f"{n}_x_px", f"{n}_y_px"]].to_numpy(dtype=float)
            for n in COCO_KEYPOINT_NAMES
        ],
        axis=1,
    )

    def g(name):
        return points[:, JOINT[name]]

    def dist(a, b):
        return np.linalg.norm(a - b, axis=1) / torso

    def speed(p):
        return np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1)] / torso

    shoulder = (g("left_shoulder") + g("right_shoulder")) / 2
    hip = (g("left_hip") + g("right_hip")) / 2
    torso = np.nanmedian(np.linalg.norm(shoulder - hip, axis=1))

    columns = []
    for base, tip in [("shoulder", "wrist"), ("hip", "ankle"),
                      ("shoulder", "elbow"), ("hip", "knee")]:
        columns += [
            dist(g(f"left_{tip}"), g(f"left_{base}")),
            dist(g(f"right_{tip}"), g(f"right_{base}")),
        ]
    columns += [
        dist(g("left_wrist"), g("right_wrist")),
        dist(g("left_ankle"), g("right_ankle")),
        np.abs(g("left_ankle")[:, 1] - g("right_ankle")[:, 1]) / torso,
        np.abs(g("left_wrist")[:, 1] - g("right_wrist")[:, 1]) / torso,
        dist(g("nose"), hip),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 전부 NaN인 프레임
        width = np.nanmax(points[:, :, 0], 1) - np.nanmin(points[:, :, 0], 1)
        height = np.nanmax(points[:, :, 1], 1) - np.nanmin(points[:, :, 1], 1)
    columns.append(height / (width + 1e-6))
    for name in ("left_wrist", "right_wrist", "left_ankle", "right_ankle", "nose"):
        columns.append(speed(g(name) - hip))
    columns.append(speed(hip))

    features = np.stack(columns, axis=1)
    # 좌우 쌍 정렬 (팔/다리 뻗은 거리 4쌍, 손목 속도, 발목 속도)
    for a, b in [(0, 1), (2, 3), (4, 5), (6, 7), (14, 15), (16, 17)]:
        low = np.minimum(features[:, a], features[:, b])
        high = np.maximum(features[:, a], features[:, b])
        features[:, a], features[:, b] = low, high
    return features


def window_stats(rows: np.ndarray) -> np.ndarray:
    velocity = np.abs(np.diff(rows, axis=0))
    return np.r_[
        rows.mean(0), rows.std(0), rows.min(0), rows.max(0), velocity.mean(0),
    ]


# ----------------------------------------------------
# hybrid: 영상 전체 요약 Feature (Clip 모델용)
#   2초 창은 대부분 "누워 있는 끝 자세"만 보여서 눕기/넘어지기/쪼그려 빼내기가
#   헷갈립니다. 영상 전체를 10구간으로 나눈 평균 자세 흐름과, 엉덩이·코·어깨가
#   가장 빨리 내려가고/올라간 속도와 시점을 추가로 봅니다.
# ----------------------------------------------------
def vertical_speed_features(df: pd.DataFrame) -> np.ndarray:
    def center(a, b):
        return (
            df[[f"{a}_x_px", f"{a}_y_px"]].to_numpy(dtype=float)
            + df[[f"{b}_x_px", f"{b}_y_px"]].to_numpy(dtype=float)
        ) / 2

    shoulder = center("left_shoulder", "right_shoulder")
    hip = center("left_hip", "right_hip")
    torso = np.nanmedian(np.linalg.norm(shoulder - hip, axis=1))
    nose_y = df["nose_y_px"].to_numpy(dtype=float)
    # 화면 아래 방향(+)으로의 세로 속도, 몸통 길이/프레임
    return np.stack(
        [np.r_[0.0, np.diff(y)] / torso for y in (hip[:, 1], nose_y, shoulder[:, 1])],
        axis=1,
    )


def clip_descriptor(frames: np.ndarray, vertical: np.ndarray, parts=10) -> np.ndarray:
    data = np.c_[frames, vertical]
    timeline = [part.mean(0) for part in np.array_split(data, parts)]
    smooth = pd.DataFrame(vertical).rolling(5, min_periods=1).mean().to_numpy()
    return np.r_[
        np.concatenate(timeline),
        data.std(0), data.min(0), data.max(0),
        smooth.max(0), smooth.min(0), np.argmax(smooth, 0) / len(smooth),
    ]


# ----------------------------------------------------
# 실시간 예측기: 사람 한 명의 최근 관절좌표 → 동작 확률
# ----------------------------------------------------
# 실시간 카메라는 FPS가 학습 영상(약 60fps)과 다르고 일정하지도 않습니다.
# 속도 Feature는 "프레임당 이동량"이라 FPS가 다르면 값이 달라지므로,
# 최근 window_sec 구간을 학습 FPS 간격으로 다시 보간한 뒤 Feature를 계산합니다.
class MotionCaptureActionPredictor:
    def __init__(self, model, info: dict, labels: dict[str, str] | None = None):
        if info.get("window_feature", info.get("feature")) != "robust2":
            raise ValueError("robust2 창 모델만 실시간 예측에 쓸 수 있습니다.")
        self.model = model
        self.window_seconds = float(info["window_sec"])
        self.fps = float(info.get("fps", 59.88))
        self.min_valid_frames = int(info.get("min_valid_frames", 10))
        self.confidence_threshold = float(info.get("confidence_threshold", 0.4))
        self.classes = [str(c) for c in model.classes_] if model is not None else []
        self.labels = labels or {}

    @classmethod
    def load(
        cls,
        model_dir,
        action_file=None,
        model_file="realtime_model.joblib",
        info_file="realtime_info.json",
    ) -> "MotionCaptureActionPredictor":
        import csv
        import json
        from pathlib import Path

        import joblib

        model_dir = Path(model_dir)
        info = json.loads((model_dir / info_file).read_text(encoding="utf-8"))
        labels = {}
        if action_file is not None and Path(action_file).exists():
            with Path(action_file).open("r", encoding="utf-8-sig") as file:
                labels = {
                    row["action_id"]: row["action_label"]
                    for row in csv.DictReader(file)
                }
        return cls(joblib.load(model_dir / model_file), info, labels)

    def label(self, action_id: str) -> str:
        return self.labels.get(action_id, action_id)

    def resample(self, times, keypoints) -> pd.DataFrame | None:
        """(N,) 시각, (N,17,2) 관절 → 학습 FPS 간격의 최근 window_sec 구간 DataFrame."""
        times = np.asarray(times, dtype=float)
        keypoints = np.asarray(keypoints, dtype=float)
        if len(times) < 2 or keypoints.shape[1:] != (17, 2):
            return None
        steps = int(round(self.window_seconds * self.fps))
        end = times[-1]
        target = end - (steps - 1 - np.arange(steps)) / self.fps
        target = target[target >= times[0]]
        if len(target) < self.min_valid_frames:
            return None
        flat = keypoints.reshape(len(times), -1)
        columns = {}
        for j, name in enumerate(COCO_KEYPOINT_NAMES):
            columns[f"{name}_x_px"] = np.interp(target, times, flat[:, 2 * j])
            columns[f"{name}_y_px"] = np.interp(target, times, flat[:, 2 * j + 1])
        return pd.DataFrame(columns)

    def feature_vector(self, times, keypoints) -> np.ndarray | None:
        df = self.resample(times, keypoints)
        if df is None:
            return None
        rows = np.c_[robust_frame_features(df), extended_frame_features(df)]
        rows = rows[np.isfinite(rows).all(axis=1)]
        if len(rows) < self.min_valid_frames:
            return None
        return window_stats(rows)

    def predict_proba(self, times, keypoints) -> np.ndarray | None:
        vector = self.feature_vector(times, keypoints)
        if vector is None:
            return None
        return self.model.predict_proba(vector.reshape(1, -1))[0]


def valid_keypoints(keypoints) -> np.ndarray | None:
    """YOLO keypoints.xy 한 사람분 (17,2). 어깨·엉덩이가 없으면 None."""
    points = np.asarray(keypoints, dtype=np.float32)
    if points.shape != (17, 2) or not np.all(np.isfinite(points)):
        return None
    core = [JOINT[n] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    if np.any(np.all(points[core] == 0, axis=1)):
        return None
    return points.copy()


# ----------------------------------------------------
# 실시간 재생 흉내 (학습 데이터 생성·평가 공용)
# ----------------------------------------------------
def simulate_camera(times, valid, camera_fps: float, rng) -> np.ndarray:
    """60fps 영상에서 camera_fps(±20% 흔들림)로 받은 것처럼 프레임 index를 고릅니다."""
    picked, t = [], float(times[0])
    while t <= times[-1]:
        picked.append(t)
        t += (1.0 / camera_fps) * rng.uniform(0.8, 1.2)
    index = np.unique(np.searchsorted(times, picked).clip(0, len(times) - 1))
    return index[np.asarray(valid)[index]]


def realtime_windows(
    predictor: MotionCaptureActionPredictor,
    times,
    keypoints,
    interval_sec: float = 0.25,
    min_fill_ratio: float = 0.8,
):
    """실시간 Buffer처럼 최근 window_sec 구간을 interval_sec마다 Feature로 만듭니다."""
    times = np.asarray(times, dtype=float)
    out, last = [], -1e9
    for i in range(len(times)):
        if times[i] - last < interval_sec:
            continue
        sel = (times > times[i] - predictor.window_seconds) & (times <= times[i])
        if times[i] - times[sel][0] < min_fill_ratio * predictor.window_seconds:
            continue
        vector = predictor.feature_vector(times[sel], keypoints[sel])
        if vector is not None:
            out.append(vector)
            last = times[i]
    return out
