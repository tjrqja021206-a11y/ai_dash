"""영상 → YOLO Pose + ByteTrack → Track ID별 관절 시계열.

학습 데이터(약 60fps)와 같은 시간 간격으로 다시 보간(resample)해서
학습 때와 같은 형태의 DataFrame(<관절>_x_px, <관절>_y_px)을 만듭니다.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.motion_capture_features import valid_keypoints
from src.pose_utils import COCO_KEYPOINT_NAMES


class VideoReadError(Exception):
    """영상 파일을 열 수 없거나 프레임이 없을 때."""


@dataclass
class VideoMeta:
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int

    def as_dict(self) -> dict:
        return {
            "fps": round(self.fps, 2),
            "frame_count": self.frame_count,
            "duration_sec": round(self.duration_sec, 2),
            "width": self.width,
            "height": self.height,
        }


@dataclass
class TrackData:
    track_id: int
    times: list[float] = field(default_factory=list)
    keypoints: list[np.ndarray] = field(default_factory=list)   # (17, 2) px
    confidences: list[float] = field(default_factory=list)      # 관절 신뢰도 평균

    @property
    def span(self) -> float:
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0


def read_video_meta(path: Path) -> VideoMeta:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoReadError("영상 파일을 열 수 없습니다. (손상되었거나 지원하지 않는 형식)")
    ok, frame = capture.read()
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if not ok or frame is None:
        raise VideoReadError("영상에서 프레임을 읽지 못했습니다.")
    if not 1.0 <= fps <= 240.0:
        fps = 30.0  # 메타데이터가 없는 영상은 30fps로 가정
    height, width = frame.shape[:2]
    return VideoMeta(fps, count, count / fps if count else 0.0, width, height)


class PosePipeline:
    def __init__(self, pose_model_path: Path, tracker_path: Path, project: dict):
        from ultralytics import YOLO  # import가 무거우므로 생성 시점에 로드

        self.model = YOLO(str(pose_model_path))
        self.tracker = str(tracker_path)
        self.project = project
        self.device = self._device(project.get("device", "auto"))
        # YOLO Tracker 상태는 모델 객체에 붙으므로 동시에 두 영상을 처리하지 않게 잠급니다.
        self._lock = threading.Lock()

    @staticmethod
    def _device(requested: str):
        if requested != "auto":
            return requested
        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def extract_tracks(self, path: Path, meta: VideoMeta) -> tuple[dict[int, TrackData], int]:
        """Track ID별 관절 시계열과 실제로 읽은 프레임 수를 반환합니다."""
        min_box_h = float(self.project.get("min_box_height_ratio", 0.0)) * meta.height
        tracks: dict[int, TrackData] = {}
        frames = 0
        with self._lock:
            # persist=False + stream=True: 요청마다 Tracker를 새로 만들어
            # 이전 영상의 Track ID가 섞이지 않습니다.
            results = self.model.track(
                source=str(path),
                stream=True,
                persist=False,
                tracker=self.tracker,
                conf=float(self.project.get("person_confidence", 0.25)),
                imgsz=int(self.project.get("image_size", 640)),
                device=self.device,
                verbose=False,
            )
            for index, result in enumerate(results):
                frames += 1
                boxes, kps = result.boxes, result.keypoints
                if boxes is None or kps is None or boxes.id is None:
                    continue
                t = index / meta.fps
                ids = boxes.id.int().cpu().tolist()
                xyxy = boxes.xyxy.cpu().numpy()
                xy = kps.xy.cpu().numpy()
                conf = kps.conf.cpu().numpy() if kps.conf is not None else None
                for i, track_id in enumerate(ids):
                    if xyxy[i, 3] - xyxy[i, 1] < min_box_h:
                        continue  # 너무 작은(멀리 있는) 사람은 무시
                    points = valid_keypoints(xy[i])
                    if points is None:
                        continue  # 어깨·엉덩이가 안 보이면 Feature를 만들 수 없음
                    track = tracks.setdefault(track_id, TrackData(track_id))
                    track.times.append(t)
                    track.keypoints.append(points)
                    track.confidences.append(
                        float(conf[i].mean()) if conf is not None else 0.0
                    )
        return tracks, frames


def resample_track(track: TrackData, fps: float, max_gap: float) -> pd.DataFrame:
    """트랙을 학습 FPS 간격으로 보간합니다.

    실제 검출 사이 간격이 max_gap보다 큰 구간(사람이 가려졌던 구간)은 NaN으로 두어
    학습 때처럼 Feature 계산에서 제외되게 합니다.
    """
    times = np.asarray(track.times, dtype=float)
    points = np.asarray(track.keypoints, dtype=float).reshape(len(times), -1)
    target = np.arange(times[0], times[-1] + 1e-9, 1.0 / fps)

    right = np.searchsorted(times, target).clip(1, len(times) - 1)
    gap = times[right] - times[right - 1]
    invalid = gap > max_gap

    columns = {}
    for j, name in enumerate(COCO_KEYPOINT_NAMES):
        for axis, k in (("x", 2 * j), ("y", 2 * j + 1)):
            values = np.interp(target, times, points[:, k])
            values[invalid] = np.nan
            columns[f"{name}_{axis}_px"] = values
    return pd.DataFrame(columns)
