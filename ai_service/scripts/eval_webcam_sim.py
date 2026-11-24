"""IP Webcam 조건을 재현해 동작 인식 성능을 평가합니다 (fastapi 컨테이너의 /app에서 실행).

사용법:
    docker compose cp <정답 영상 폴더> fastapi:/tmp/e      # 파일명에 _A001_ 같은 정답 ID 포함
    docker compose exec fastapi python scripts/eval_webcam_sim.py - orig,pad,fps15clean,webcam

1) 영상마다 가장 오래 보인 사람의 관절 트랙을 한 번 추출해 /tmp/tracks.pkl에 저장 (다시 돌릴 때 재사용)
2) 조건
   orig       : 원본
   pad        : 동작 앞뒤에 정지(미세 흔들림) 구간을 붙여 8초로 (웹캠 8초 수집 재현)
   fps15clean : 15fps ±20% 간격으로만 다시 뽑기
   noise60    : 관절 위치 흔들림만
   webcam     : pad + 15fps + 흔들림
3) 현재 서비스(project.json 설정)의 정확도·평균 신뢰도·불확실(<0.4) 비율 출력
   두 번째 인자로 METHODS dict를 정의한 파이썬 파일을 주면 다른 방법도 함께 비교
"""

import copy
import os
import glob
import pickle
import re
import sys
from pathlib import Path

import numpy as np

from services.action_inference import ActionInferenceService
from services.model_loader import load_bundle
from services.pose_pipeline import TrackData, read_video_meta

CACHE = Path("/tmp/tracks.pkl")
svc = ActionInferenceService(load_bundle())


def load_tracks():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    data = []
    for v in sorted(glob.glob(os.environ.get("EVAL_VIDEOS", "/tmp/e") + "/*.mp4")):
        meta = read_video_meta(Path(v))
        tracks, _ = svc.pipeline.extract_tracks(Path(v), meta)
        main = max(tracks.values(), key=lambda t: len(t.times))
        data.append((re.search(r"_(A\d{3})_", v).group(1), Path(v).name, main))
        print("extracted", Path(v).name, len(main.times), flush=True)
    CACHE.write_bytes(pickle.dumps(data))
    return data


def webcam(track: TrackData, rng, pad=2.2, fps=15.0, noise=0.004, total=8.0) -> TrackData:
    """앞뒤로 정지(미세 흔들림) 구간을 붙이고 15fps±20%로 다시 뽑은 트랙."""
    times = np.asarray(track.times) - track.times[0]
    pts = np.asarray(track.keypoints, float)
    torso = np.nanmedian(np.linalg.norm(pts[:, [5, 6]].mean(1) - pts[:, [11, 12]].mean(1), axis=1))
    dt = 1 / 60
    span = times[-1]
    pre = max(0.5, min(pad, (total - span) / 2))
    post = max(0.5, total - span - pre)
    t_pre = np.arange(0, pre, dt)
    t_post = np.arange(0, post, dt)
    full_t = np.r_[t_pre, pre + times, pre + span + dt + t_post]
    sway = lambda n, base: base + rng.normal(0, noise * torso, (n, 17, 2)).cumsum(0) * 0.15
    full_p = np.r_[sway(len(t_pre), pts[0]), pts, sway(len(t_post), pts[-1])]
    # 15fps ± 20% 간격으로 샘플링
    picked, t = [], 0.0
    while t <= full_t[-1]:
        picked.append(t)
        t += (1 / fps) * rng.uniform(0.8, 1.2)
    idx = np.unique(np.searchsorted(full_t, picked).clip(0, len(full_t) - 1))
    p = full_p[idx] + rng.normal(0, noise * torso, (len(idx), 17, 2))
    out = TrackData(track.track_id)
    out.times = list(full_t[idx])
    out.keypoints = list(p.astype(np.float32))
    out.confidences = [0.9] * len(idx)
    out.frame_indices = list(range(len(idx)))
    return out


CONDITIONS = {
    "webcam": {},                                            # 정지 구간 + 15fps + 흔들림 (8초)
    "pad": {"fps": 60.0, "noise": 0.0},                       # 정지 구간만
    "fps15": {"total": 0.0, "pad": 0.0},                      # 15fps·흔들림만 (정지 구간 최소)
    "fps15clean": {"total": 0.0, "pad": 0.0, "noise": 0.0},   # 15fps만
    "noise60": {"total": 0.0, "pad": 0.0, "fps": 60.0},       # 관절 흔들림만
}


def evaluate(data, methods, conditions, seeds=3):
    for cond in conditions:
        for name, fn in methods.items():
            correct = total = 0
            confs, right_conf = [], []
            for seed in range(seeds if cond != "orig" else 1):
                rng = np.random.default_rng(seed)
                for gt, vname, track in data:
                    t = track if cond == "orig" else webcam(track, rng, **CONDITIONS[cond])
                    r = fn(copy.deepcopy(t))
                    pred = r["action_id"] if r else None
                    total += 1
                    correct += pred == gt
                    confs.append(r["confidence"] if r else 0)
            print(f"{cond:7s} {name:22s} acc {correct}/{total} = {correct/total:.3f}  "
                  f"mean_conf {np.mean(confs):.3f}  uncertain(<0.4) {np.mean(np.array(confs) < 0.4):.2f}",
                  flush=True)


if __name__ == "__main__":
    data = load_tracks()
    methods = {"현재 서비스": svc.predict_track}
    extra = sys.argv[1] if len(sys.argv) > 1 else None
    if extra and extra != "-":
        ns = {"svc": svc, "np": np, "TrackData": TrackData}
        exec(Path(extra).read_text(), ns)
        methods.update(ns["METHODS"])
    conds = (sys.argv[2] if len(sys.argv) > 2 else "orig,webcam").split(",")
    evaluate(data, methods, conds)
