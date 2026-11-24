"""Track ID별 관절 시계열 → Random Forest(hybrid) 예측.

학습 스크립트(subject10_pose_action/scripts/48_test_motion_capture_actions.py)의
clip_vectors()·run_fold()와 같은 순서로 계산합니다.

  1. 트랙 전체에서 Frame Feature 35개(robust 15 + extended 20)를 계산
  2. 2초 창을 0.25초 간격으로 잘라 창마다 5가지 통계(175차원) → 창 모델 확률
  3. 트랙 전체 요약(503차원) → Clip 모델 확률
  4. hybrid 확률 = (창 확률 평균 + Clip 확률) / 2  → 가장 큰 동작이 예측값
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from services.model_loader import ModelBundle
from services.pose_pipeline import (
    PosePipeline,
    TrackData,
    read_video_meta,
    resample_track,
)
from src.motion_capture_features import (
    clip_descriptor,
    extended_frame_features,
    robust_frame_features,
    vertical_speed_features,
    window_stats,
)


logger = logging.getLogger("uvicorn.error")


class VideoTooShortError(Exception):
    """영상이 너무 짧아 동작을 판단할 수 없을 때."""


class ActionInferenceService:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.pipeline = PosePipeline(bundle.pose_model_path, bundle.tracker_path, bundle.project)
        info, project = bundle.info, bundle.project
        self.fps = float(info["fps"])
        self.window = int(round(float(info["window_sec"]) * self.fps))
        self.stride = max(1, int(round(float(info["stride_sec"]) * self.fps)))
        self.min_valid = int(info["min_valid_frames"])
        self.min_track_sec = float(project.get("min_track_seconds", 1.0))
        self.max_gap = float(project.get("max_gap_seconds", 0.25))
        self.uncertain = float(project.get("uncertain_threshold", 0.4))

    # ------------------------------------------------
    # Feature
    # ------------------------------------------------
    def track_vectors(self, track: TrackData):
        """(창 Feature 목록, Clip Feature 또는 None)"""
        df = resample_track(track, self.fps, self.max_gap)
        frame = np.c_[robust_frame_features(df), extended_frame_features(df)]
        valid = np.isfinite(frame).all(axis=1)

        windows = []
        n = len(df)
        starts = range(0, max(1, n - self.window + 1), self.stride)
        for start in starts:
            end = min(n, start + self.window)
            rows = frame[start:end][valid[start:end]]
            if len(rows) >= self.min_valid:
                windows.append(window_stats(rows))

        vertical = vertical_speed_features(df)
        whole = valid & np.isfinite(vertical).all(axis=1)
        clip = (
            clip_descriptor(frame[whole], vertical[whole])
            if whole.sum() >= self.min_valid else None
        )
        return windows, clip

    def predict_track(self, track: TrackData) -> dict | None:
        windows, clip = self.track_vectors(track)
        if not windows or clip is None:
            return None
        window_proba = self.bundle.window_model.predict_proba(np.stack(windows)).mean(0)
        clip_proba = self.bundle.clip_model.predict_proba(clip.reshape(1, -1))[0]
        proba = (window_proba + clip_proba) / 2.0

        classes = self.bundle.classes
        order = np.argsort(proba)[::-1]
        best = classes[order[0]]
        confidence = float(proba[order[0]])
        action = self.bundle.actions[best]
        return {
            "track_id": int(track.track_id),
            "action_id": best,
            "action_label": action["action_label"],
            "action_name": action["description"],
            "confidence": round(confidence, 4),
            "is_uncertain": confidence < self.uncertain,
            "top3": [
                {
                    "action_id": classes[i],
                    "action_label": self.bundle.actions[classes[i]]["action_label"],
                    "probability": round(float(proba[i]), 4),
                }
                for i in order[:3]
            ],
            "window_count": len(windows),
            "frame_count": len(track.times),
            "duration_sec": round(track.span, 2),
            "mean_keypoint_conf": round(float(np.mean(track.confidences)), 3),
        }

    # ------------------------------------------------
    # 영상 한 건 처리
    # ------------------------------------------------
    def analyze_video(self, path: Path, overlay_dir: Path | None = None) -> dict:
        """overlay_dir를 주면 Skeleton Overlay 영상도 만들어 응답의 overlay에 정보를 넣습니다."""
        started = time.perf_counter()
        meta = read_video_meta(path)
        if 0 < meta.duration_sec < self.min_track_sec:
            raise VideoTooShortError(
                f"영상이 너무 짧습니다. ({meta.duration_sec:.1f}초 < {self.min_track_sec}초)"
            )

        tracks, frames_read = self.pipeline.extract_tracks(path, meta)
        if frames_read == 0:
            raise VideoTooShortError("영상에서 프레임을 읽지 못했습니다.")

        predictions, skipped = [], []
        for track in sorted(tracks.values(), key=lambda t: t.track_id):
            if track.span < self.min_track_sec or len(track.times) < self.min_valid:
                skipped.append(track.track_id)
                continue
            result = self.predict_track(track)
            if result is None:
                skipped.append(track.track_id)
            else:
                predictions.append(result)

        if predictions:
            status, message = "ok", f"{len(predictions)}명의 동작을 예측했습니다."
        elif tracks:
            status = "no_valid_person"
            message = "사람은 검출됐지만 동작을 판단할 만큼 오래·선명하게 보이지 않았습니다."
        else:
            status, message = "no_person", "영상에서 사람을 찾지 못했습니다."

        inference_ms = int((time.perf_counter() - started) * 1000)
        overlay, overlay_error = None, None
        if overlay_dir is not None and tracks:
            from services.overlay import render_overlay

            try:
                info = render_overlay(
                    path, meta, tracks,
                    {p["track_id"]: p for p in predictions},
                    overlay_dir,
                    analysis_stride=self.pipeline.frame_stride(meta.fps),
                )
                overlay = {k: v for k, v in info.items() if k != "path"}
                overlay["url"] = f"/overlay/{info['id']}"
            except Exception as error:  # Overlay 실패가 예측 결과를 막지 않게 함
                logger.exception("Overlay 생성 실패")
                overlay_error = f"Overlay 영상을 만들지 못했습니다: {error}"

        return {
            "status": status,
            "message": message,
            "video": {**meta.as_dict(), "frames_read": frames_read},
            "predictions": predictions,
            "skipped_track_ids": skipped,
            "uncertain_threshold": self.uncertain,
            "processing_ms": inference_ms,
            "overlay": overlay,
            "overlay_error": overlay_error,
            "total_ms": int((time.perf_counter() - started) * 1000),
        }
