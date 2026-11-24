"""Celery 작업: 영상 확보 → FastAPI 추론 → Django ORM 저장.

상태 흐름: PENDING → STARTED → SUCCESS 또는 FAILURE
"""

import logging
from pathlib import Path

from celery import shared_task
from django.conf import settings
from django.core.files import File
from django.db import transaction
from django.utils import timezone

from .camera_capture import CameraError, capture_clip
from .models import AnalysisJob, PersonPrediction
from .services import AIServiceError, predict_video

logger = logging.getLogger(__name__)


def _set_stage(job: AnalysisJob, stage: str) -> None:
    job.stage = stage
    job.save(update_fields=["stage"])


def _obtain_video(job: AnalysisJob) -> Path:
    """업로드 영상이면 그 경로, IP Webcam이면 지정 시간만큼 수집해 저장한 경로."""
    if job.input_type == AnalysisJob.InputType.UPLOAD:
        if not job.video:
            raise CameraError("업로드된 영상이 없습니다.")
        return Path(job.video.path)

    url = settings.ALLOWED_CAMERAS.get(job.camera_name)
    if not url:
        raise CameraError(f"허용되지 않은 카메라입니다: {job.camera_name}")
    _set_stage(job, f"영상 수집 중 ({job.capture_seconds}초)")
    temp = Path(settings.MEDIA_ROOT) / "captures" / f"job_{job.pk}_tmp.mp4"
    capture_clip(url, job.capture_seconds, temp, settings.CAMERA_OPEN_TIMEOUT_SECONDS)
    # 수집한 영상도 업로드와 같은 방식(FileField)으로 보관 → 결과 화면에서 재생 가능
    with temp.open("rb") as file:
        job.video.save(f"camera_job_{job.pk}.mp4", File(file), save=True)
    temp.unlink(missing_ok=True)
    return Path(job.video.path)


@shared_task(bind=True)
def run_analysis(self, job_id: int) -> str:
    try:
        job = AnalysisJob.objects.get(pk=job_id)
    except AnalysisJob.DoesNotExist:
        logger.warning("작업 #%s가 없습니다 (삭제됨)", job_id)
        return "missing"

    job.status = AnalysisJob.Status.STARTED
    job.started_at = timezone.now()
    job.celery_task_id = self.request.id or job.celery_task_id
    job.stage = "영상 준비"
    job.save(update_fields=["status", "started_at", "celery_task_id", "stage"])
    logger.info("작업 #%s 시작 (%s)", job.pk, job.input_type)

    try:
        video_path = _obtain_video(job)
        _set_stage(job, "AI 분석 중")
        result = predict_video(video_path)

        with transaction.atomic():
            job.predictions.all().delete()  # 재실행 시 중복 방지
            PersonPrediction.objects.bulk_create(
                [
                    PersonPrediction(
                        job=job,
                        track_id=p["track_id"],
                        action_id=p["action_id"],
                        action_label=p["action_label"],
                        confidence=p["confidence"],
                        is_uncertain=p["confidence"] < settings.UNCERTAIN_THRESHOLD,
                        top3=p.get("top3", []),
                        window_count=p.get("window_count", 0),
                        frame_count=p.get("frame_count", 0),
                        duration_sec=p.get("duration_sec", 0),
                        mean_keypoint_conf=p.get("mean_keypoint_conf"),
                    )
                    for p in result.get("predictions", [])
                ]
            )
            video = result.get("video", {})
            job.status = AnalysisJob.Status.SUCCESS
            job.stage = "완료"
            job.result_status = result.get("status", "")
            job.result_message = result.get("message", "")[:200]
            job.video_duration_sec = video.get("duration_sec")
            job.video_fps = video.get("fps")
            job.processing_ms = result.get("processing_ms")
            job.error_message = ""
            job.completed_at = timezone.now()
            job.save()
        logger.info("작업 #%s 완료: %s", job.pk, job.result_message)
        return job.status

    except (CameraError, AIServiceError, OSError) as error:
        message = str(error)
    except Exception as error:  # 예상 못 한 오류도 FAILURE로 남김
        logger.exception("작업 #%s 예기치 않은 오류", job_id)
        message = f"예기치 않은 오류: {error}"

    AnalysisJob.objects.filter(pk=job_id).update(
        status=AnalysisJob.Status.FAILURE,
        stage="실패",
        error_message=message,
        completed_at=timezone.now(),
    )
    logger.warning("작업 #%s 실패: %s", job_id, message)
    return AnalysisJob.Status.FAILURE
