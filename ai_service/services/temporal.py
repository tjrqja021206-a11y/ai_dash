"""동작 앞뒤의 긴 정지 구간 줄이기 (IP Webcam 8초 수집 대응).

학습 영상은 동작 하나가 담긴 클립이라 동작 전후 정지 구간이 짧습니다
(34개 영상 측정: 앞 중앙값 0.13초, 뒤 0.31초). 웹캠은 정해진 시간(기본 8초)을 수집하므로
동작 앞뒤로 가만히 선 구간이 몇 초씩 붙고, 창 확률을 평균할 때 이 구간이 섞여 신뢰도가 떨어집니다.

움직임 곡선(관절 이동량 / 몸통 길이)에서 '움직임 최대값의 motion_rel 이상'인 첫~마지막 시점을 찾고,
그 앞뒤로 pre/post초만 남긴 트랙을 만듭니다. 추론은 전체 트랙과 줄인 트랙의 확률을 함께 씁니다
(action_inference.predict_track).
"""

from __future__ import annotations

import copy

import numpy as np

from services.pose_pipeline import TrackData

# 얼굴 일부 대신 몸 전체 움직임을 보도록: 코 + 어깨·팔꿈치·손목·골반·무릎·발목
MOTION_JOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def motion_curve(track: TrackData, fps: float = 60.0, smooth_sec: float = 0.25):
    """일정 간격(fps) 시각 배열과 그 시점의 움직임 크기(몸통 길이 기준, smooth_sec 이동평균)."""
    t = np.asarray(track.times, float)
    p = np.asarray(track.keypoints, float)
    torso = np.nanmedian(np.linalg.norm(p[:, [5, 6]].mean(1) - p[:, [11, 12]].mean(1), axis=1))
    grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / fps)
    flat = p[:, MOTION_JOINTS].reshape(len(t), -1)
    g = np.stack([np.interp(grid, t, flat[:, k]) for k in range(flat.shape[1])], axis=1)
    g = g.reshape(len(grid), len(MOTION_JOINTS), 2)
    motion = np.r_[0.0, np.linalg.norm(np.diff(g, axis=0), axis=2).mean(1)] / max(torso, 1e-6)
    k = max(1, int(smooth_sec * fps))
    return grid, np.convolve(motion, np.ones(k) / k, mode="same")


def crop(track: TrackData, t0: float, t1: float) -> TrackData:
    keep = [t0 <= t <= t1 for t in track.times]
    out = copy.copy(track)
    for name in ("times", "keypoints", "confidences", "frame_indices"):
        setattr(out, name, [x for x, k in zip(getattr(track, name), keep) if k])
    return out


def trim_static(
    track: TrackData, pre: float = 0.6, post: float = 0.8, motion_rel: float = 0.2, min_frames: int = 10
) -> tuple[TrackData, float]:
    """(앞뒤 정지 구간을 줄인 트랙, 잘라 낸 초). 줄일 것이 없으면 원래 트랙과 0."""
    if len(track.times) < 2:
        return track, 0.0
    grid, motion = motion_curve(track)
    if not np.isfinite(motion).all() or motion.max() <= 0:
        return track, 0.0
    active = np.where(motion >= motion_rel * motion.max())[0]
    t0 = max(grid[0], grid[active[0]] - pre)
    t1 = min(grid[-1], grid[active[-1]] + post)
    removed = (t0 - grid[0]) + (grid[-1] - t1)
    if removed < 0.05:
        return track, 0.0
    out = crop(track, t0, t1)
    if len(out.times) < min_frames:
        return track, 0.0
    return out, float(removed)
