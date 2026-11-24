"""허용된 IP Webcam에서 정해진 시간만큼 프레임을 읽어 MP4로 저장합니다.

- Celery Worker 컨테이너에서 실행되므로, Worker가 카메라 주소에 접근할 수 있어야 합니다.
- 저장 FPS는 '실제로 받은 프레임 수 / 실제 수집 시간'으로 계산합니다.
  (휴대폰 카메라는 FPS가 일정하지 않으므로, 영상 시간축이 실제 시간과 맞아야
   AI 서버의 속도 Feature가 학습 때와 같은 의미를 가집니다.)
- 프레임은 받는 즉시 회전·축소 후 JPEG로 압축해 메모리에 둡니다.
  (1080p 원본 480장을 그대로 모으면 약 3GB → JPEG로 약 50MB)
"""

import logging
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MIN_FPS = 5.0          # 이보다 적게 들어오면 동작 판단이 어려움
MAX_READ_FAILURES = 30  # 연속으로 프레임을 못 읽으면 연결 끊김으로 판단
JPEG_QUALITY = 92

_ROTATE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class CameraError(Exception):
    """카메라 연결 실패·끊김·프레임 부족."""


def prepare_frame(frame: np.ndarray, rotate: int = 0, max_side: int = 0) -> np.ndarray:
    """카메라 프레임을 시계 방향으로 rotate도 돌리고, 긴 변이 max_side를 넘으면 줄입니다."""
    if rotate in _ROTATE:
        frame = cv2.rotate(frame, _ROTATE[rotate])
    height, width = frame.shape[:2]
    if max_side and max(height, width) > max_side:
        scale = max_side / max(height, width)
        size = (int(width * scale) // 2 * 2, int(height * scale) // 2 * 2)
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return frame


def snapshot_url(stream_url: str) -> str:
    """IP Webcam 앱의 영상 주소(…/video)에서 정지 화면 주소(…/shot.jpg)를 만듭니다."""
    parts = urlsplit(stream_url)
    return urlunsplit((parts.scheme, parts.netloc, "/shot.jpg", "", ""))


def capture_clip(
    url: str,
    seconds: int,
    output: Path,
    open_timeout: int = 10,
    rotate: int = 0,
    max_side: int = 1280,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(
        url,
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout * 1000,
         cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
    )
    if not capture.isOpened():
        raise CameraError("카메라에 연결할 수 없습니다. IP Webcam 앱 실행·같은 네트워크인지 확인하세요.")

    encoded: list[bytes] = []
    size = None
    failures = 0
    started = time.monotonic()
    try:
        while time.monotonic() - started < seconds:
            ok, frame = capture.read()
            if not ok or frame is None:
                failures += 1
                if failures >= MAX_READ_FAILURES:
                    raise CameraError("수집 도중 카메라 연결이 끊겼습니다.")
                time.sleep(0.05)
                continue
            failures = 0
            frame = prepare_frame(frame, rotate, max_side)
            if size is None:
                size = (frame.shape[1], frame.shape[0])
            elif (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)  # 도중에 해상도가 바뀌어도 같은 크기로 저장
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                encoded.append(jpeg.tobytes())
    finally:
        capture.release()

    elapsed = time.monotonic() - started
    fps = len(encoded) / elapsed if elapsed > 0 else 0.0
    if fps < MIN_FPS:
        raise CameraError(
            f"받은 프레임이 너무 적습니다. ({len(encoded)}장 / {elapsed:.1f}초 = {fps:.1f}fps)"
        )

    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for jpeg in encoded:
        writer.write(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR))
    writer.release()

    logger.info(
        "카메라 수집 완료: %d장, %.1f초, %.1ffps, %dx%d, 회전 %d° → %s",
        len(encoded), elapsed, fps, size[0], size[1], rotate, output,
    )
    return {"frames": len(encoded), "seconds": round(elapsed, 2), "fps": round(fps, 2)}
