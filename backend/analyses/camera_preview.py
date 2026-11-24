"""IP Webcam 실시간 미리보기 (브라우저가 0.2초 간격으로 요청).

브라우저 → Django(로그인·허용 카메라 확인) → 휴대폰 /shot.jpg → 회전·축소
        → FastAPI /pose/frame(관절 그리기·촬영 자세 판정) → JPEG

- 브라우저는 카메라 주소를 모릅니다. 허용 목록의 '이름'으로만 요청합니다.
- AI 서버가 응답하지 않으면 관절 없이 카메라 화면만 보냅니다(X-Pose-Status: unknown).
"""

import logging

import cv2
import numpy as np
import requests
from django.conf import settings

from .camera_capture import CameraError, prepare_frame, snapshot_url

logger = logging.getLogger(__name__)

PREVIEW_MAX_SIDE = 640


def preview_frame(camera_name: str) -> tuple[bytes, dict]:
    """(JPEG, 헤더) — 카메라에 연결할 수 없으면 CameraError."""
    url = settings.ALLOWED_CAMERAS[camera_name]
    try:
        response = requests.get(snapshot_url(url), timeout=(2, 4))
        response.raise_for_status()
    except requests.RequestException as error:
        raise CameraError("카메라 화면을 가져올 수 없습니다. IP Webcam 앱에서 서버를 켜 주세요.") from error

    frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise CameraError("카메라가 보낸 이미지를 읽을 수 없습니다.")
    frame = prepare_frame(frame, settings.CAMERA_ROTATION.get(camera_name, 0), PREVIEW_MAX_SIDE)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise CameraError("미리보기 이미지를 만들 수 없습니다.")
    jpeg = jpeg.tobytes()

    headers = {"X-Pose-Status": "unknown", "Cache-Control": "no-store"}
    try:
        pose = requests.post(
            f"{settings.FASTAPI_URL.rstrip('/')}/pose/frame",
            files={"file": ("frame.jpg", jpeg, "image/jpeg")},
            timeout=(2, 4),
        )
        if pose.status_code == 200:
            jpeg = pose.content
            for key in ("X-Pose-Status", "X-Pose-Persons", "X-Pose-Keypoint-Conf"):
                if key in pose.headers:
                    headers[key] = pose.headers[key]
    except requests.RequestException as error:
        logger.debug("미리보기 관절 검출 생략: %s", error)
    return jpeg, headers
