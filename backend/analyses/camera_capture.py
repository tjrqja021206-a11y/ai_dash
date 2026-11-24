"""허용된 IP Webcam에서 정해진 시간만큼 프레임을 읽어 MP4로 저장합니다.

- Celery Worker 컨테이너에서 실행되므로, Worker가 카메라 주소에 접근할 수 있어야 합니다.
- 저장 FPS는 '실제로 받은 프레임 수 / 실제 수집 시간'으로 계산합니다.
  (휴대폰 카메라는 FPS가 일정하지 않으므로, 영상 시간축이 실제 시간과 맞아야
   AI 서버의 속도 Feature가 학습 때와 같은 의미를 가집니다.)
"""

import logging
import time
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

MIN_FPS = 5.0          # 이보다 적게 들어오면 동작 판단이 어려움
MAX_READ_FAILURES = 30  # 연속으로 프레임을 못 읽으면 연결 끊김으로 판단


class CameraError(Exception):
    """카메라 연결 실패·끊김·프레임 부족."""


def capture_clip(url: str, seconds: int, output: Path, open_timeout: int = 10) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(
        url,
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout * 1000,
         cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
    )
    if not capture.isOpened():
        raise CameraError("카메라에 연결할 수 없습니다. IP Webcam 앱 실행·같은 네트워크인지 확인하세요.")

    frames = []
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
            frames.append(frame)
    finally:
        capture.release()

    elapsed = time.monotonic() - started
    fps = len(frames) / elapsed if elapsed > 0 else 0.0
    if fps < MIN_FPS:
        raise CameraError(
            f"받은 프레임이 너무 적습니다. ({len(frames)}장 / {elapsed:.1f}초 = {fps:.1f}fps)"
        )

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()

    logger.info("카메라 수집 완료: %d장, %.1f초, %.1ffps → %s", len(frames), elapsed, fps, output)
    return {"frames": len(frames), "seconds": round(elapsed, 2), "fps": round(fps, 2)}
