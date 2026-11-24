"""Skeleton Overlay 결과 영상 만들기.

추론이 끝난 뒤 원본 영상을 한 번 더 읽어, 추적한 사람마다 다른 색으로
관절 뼈대와 "ID·동작·신뢰도"를 그리고 브라우저에서 바로 재생되는 H.264 MP4로 저장합니다.

- 디코딩은 추론과 같은 OpenCV(prefetch_frames) → 회전·색이 분석한 화면과 같음
- 인코딩은 PyAV(libx264): OpenCV pip 빌드에는 H.264 인코더가 없음
- 크기는 긴 변 720px, 30fps 이하로 줄여 파일을 작게 유지
"""

from __future__ import annotations

import time
import uuid
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from services.pose_pipeline import TrackData, VideoMeta, prefetch_frames
from src.pose_utils import COCO_SKELETON

MAX_SIDE = 720
MAX_FPS = 30.0
# 추적 번호마다 구분되는 색 (BGR)
PALETTE = [
    (66, 135, 245), (60, 180, 75), (245, 130, 48), (240, 50, 230),
    (70, 240, 240), (255, 225, 25), (230, 25, 75), (145, 30, 180),
]
SKIPPED_COLOR = (160, 160, 160)


def _even(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


def _lookup(track: TrackData, max_back: int) -> dict[int, np.ndarray]:
    """프레임 번호 → 관절 (17,2). 처리하지 않은 프레임은 직전 검출을 max_back장까지 이어서 씁니다."""
    table: dict[int, np.ndarray] = {}
    for index, points in zip(track.frame_indices, track.keypoints):
        for offset in range(max_back + 1):
            table.setdefault(index + offset, points)
    return table


def _label(track_id: int, prediction: dict | None) -> str:
    if prediction is None:
        return f"ID{track_id}"
    mark = "?" if prediction["is_uncertain"] else ""
    return (
        f"ID{track_id} {prediction['action_id']} {prediction['action_label']}"
        f" {prediction['confidence'] * 100:.0f}%{mark}"
    )


def _draw_person(frame, points, scale, color, label, thickness):
    pts = points * scale
    visible = ~np.all(points == 0, axis=1)  # YOLO는 신뢰도 낮은 관절을 (0,0)으로 둠
    for a, b in COCO_SKELETON:
        if visible[a] and visible[b]:
            cv2.line(frame, tuple(map(int, pts[a])), tuple(map(int, pts[b])), color,
                     thickness, cv2.LINE_AA)
    for x, y in pts[visible]:
        cv2.circle(frame, (int(x), int(y)), thickness + 1, color, -1, cv2.LINE_AA)

    # 이름표: 보이는 관절 중 가장 위쪽 위에
    top = pts[visible]
    x = int(np.clip(top[:, 0].mean(), 0, frame.shape[1] - 1))
    y = int(top[:, 1].min()) - 12
    font, size = cv2.FONT_HERSHEY_SIMPLEX, 0.45 + max(frame.shape[:2]) / 2400
    (w, h), base = cv2.getTextSize(label, font, size, 1)
    x = int(np.clip(x - w // 2, 2, frame.shape[1] - w - 2))
    y = int(np.clip(y, h + 4, frame.shape[0] - base - 2))
    cv2.rectangle(frame, (x - 3, y - h - 4), (x + w + 3, y + base + 2), color, -1)
    cv2.putText(frame, label, (x, y), font, size, (255, 255, 255), 1, cv2.LINE_AA)


def render_overlay(
    path: Path,
    meta: VideoMeta,
    tracks: dict[int, TrackData],
    predictions: dict[int, dict],
    out_dir: Path,
    analysis_stride: int = 1,
) -> dict:
    """Overlay MP4를 out_dir에 저장하고 {id, path, frames, fps, width, height, render_ms}를 반환합니다."""
    import av  # 무거운 import는 사용할 때만

    started = time.perf_counter()
    stride = 1 if meta.fps <= MAX_FPS * 1.25 else max(1, round(meta.fps / MAX_FPS))
    fps = meta.fps / stride
    scale = min(1.0, MAX_SIDE / max(meta.width, meta.height))
    width, height = _even(meta.width * scale), _even(meta.height * scale)
    thickness = max(2, round(max(width, height) / 300))

    max_back = max(stride, analysis_stride)
    people = [
        (track_id, _lookup(track, max_back), predictions.get(track_id))
        for track_id, track in sorted(tracks.items())
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_id = uuid.uuid4().hex
    out_path = out_dir / f"{overlay_id}.mp4"
    container = av.open(str(out_path), "w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"preset": "veryfast", "crf": "26"}

    written = 0
    try:
        for index, frame in prefetch_frames(path, stride):
            if frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            for track_id, table, prediction in people:
                points = table.get(index)
                if points is None:
                    continue
                color = SKIPPED_COLOR if prediction is None else PALETTE[track_id % len(PALETTE)]
                _draw_person(frame, points, scale, color, _label(track_id, prediction), thickness)
            cv2.putText(frame, f"{index / meta.fps:5.1f}s", (8, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
                container.mux(packet)
            written += 1
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    if written == 0:
        out_path.unlink(missing_ok=True)
        raise ValueError("Overlay 영상에 쓸 프레임이 없습니다.")
    return {
        "id": overlay_id,
        "path": out_path,
        "frames": written,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "render_ms": int((time.perf_counter() - started) * 1000),
    }


def cleanup_old(out_dir: Path, max_age_sec: float = 3600) -> None:
    """Worker가 받아가지 않은 오래된 Overlay 파일 정리."""
    if not out_dir.exists():
        return
    limit = time.time() - max_age_sec
    for file in out_dir.glob("*.mp4"):
        try:
            if file.stat().st_mtime < limit:
                file.unlink()
        except OSError:
            pass
