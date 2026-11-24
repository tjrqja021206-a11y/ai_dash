"""카메라 미리보기용: 사진 한 장 → YOLO Pose → 관절을 그린 JPEG + 촬영 자세 판정.

영상 분석(PosePipeline)과 다른 YOLO 객체를 써서, 분석 중에도 미리보기가 기다리지 않습니다.
판정 코드는 HTTP 헤더(ASCII)로 보내고, 한글 안내 문구는 화면(JS)에서 붙입니다.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from services.overlay import PALETTE, _draw_person

KEYPOINT_OK = 0.5
CORE = (5, 6, 11, 12)          # 어깨·골반: 없으면 Feature를 만들 수 없음
ANKLES = (15, 16)
GOOD_COLOR = (60, 180, 75)
WARN_COLOR = (0, 140, 255)


def assess(points: np.ndarray, conf: np.ndarray, box: np.ndarray, frame_h: int) -> str:
    """가장 큰 사람 한 명의 촬영 상태 코드.

    ok          : 전신이 보임
    core_missing: 어깨·골반이 안 보임 (너무 가깝거나 가려짐)
    sideways    : 몸통이 가로로 누워 보임 (카메라 회전 설정 확인)
    legs_missing: 발목이 안 보임 (조금 더 뒤로)
    too_far     : 사람이 너무 작음 (조금 더 앞으로)
    """
    if np.any(conf[list(CORE)] < KEYPOINT_OK):
        return "core_missing"
    shoulder = points[[5, 6]].mean(axis=0)
    hip = points[[11, 12]].mean(axis=0)
    dx, dy = np.abs(shoulder - hip)
    if dx > dy:
        return "sideways"
    if np.all(conf[list(ANKLES)] < KEYPOINT_OK):
        return "legs_missing"
    if (box[3] - box[1]) < 0.2 * frame_h:
        return "too_far"
    return "ok"


class LivePose:
    def __init__(self, pose_model_path: Path, device, image_size: int = 640, conf: float = 0.25):
        from ultralytics import YOLO

        self.model = YOLO(str(pose_model_path))
        self.device = device
        self.image_size = image_size
        self.conf = conf
        self._lock = threading.Lock()  # 같은 YOLO 객체를 동시에 쓰지 않게

    def annotate(self, jpeg: bytes) -> tuple[bytes, dict]:
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("이미지를 읽을 수 없습니다.")
        with self._lock:
            result = self.model.predict(
                frame, imgsz=self.image_size, conf=self.conf, device=self.device, verbose=False
            )[0]

        info = {"persons": 0, "status": "no_person", "keypoint_conf": 0.0}
        boxes, kps = result.boxes, result.keypoints
        if boxes is not None and kps is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            xy = kps.xy.cpu().numpy()
            conf = kps.conf.cpu().numpy() if kps.conf is not None else np.zeros(xy.shape[:2])
            main = int(np.argmax(xyxy[:, 3] - xyxy[:, 1]))  # 가장 크게 보이는 사람 기준으로 판정
            status = assess(xy[main], conf[main], xyxy[main], frame.shape[0])
            info = {
                "persons": len(xyxy),
                "status": status,
                "keypoint_conf": round(float(conf[main].mean()), 2),
            }
            thickness = max(2, round(max(frame.shape[:2]) / 300))
            for i in range(len(xyxy)):
                if i == main:
                    color = GOOD_COLOR if status == "ok" else WARN_COLOR
                else:
                    color = PALETTE[i % len(PALETTE)]
                label = f"{conf[i].mean() * 100:.0f}%"
                _draw_person(frame, xy[i], 1.0, color, label, thickness)

        ok, out = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError("이미지를 만들 수 없습니다.")
        return out.tobytes(), info
