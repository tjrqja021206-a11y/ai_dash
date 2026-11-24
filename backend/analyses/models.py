"""업무 테이블 2개. 회원은 Django 기본 auth_user를 사용합니다.

auth_user 1 ─── N AnalysisJob 1 ─── N PersonPrediction
"""

from django.conf import settings
from django.db import models

from .actions import ACTION_CHOICES, ACTION_NAMES


def upload_path(instance, filename):
    return f"uploads/user_{instance.user_id}/{filename}"


class AnalysisJob(models.Model):
    """사용자의 분석 요청 1건 (영상 업로드 또는 IP Webcam 수집)."""

    class InputType(models.TextChoices):
        UPLOAD = "UPLOAD", "영상 업로드"
        IP_WEBCAM = "IP_WEBCAM", "IP Webcam"

    class Status(models.TextChoices):
        PENDING = "PENDING", "대기"
        STARTED = "STARTED", "처리 중"
        SUCCESS = "SUCCESS", "완료"
        FAILURE = "FAILURE", "실패"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="analysis_jobs"
    )
    title = models.CharField("제목", max_length=100, blank=True)
    memo = models.TextField("메모", blank=True)
    input_type = models.CharField("입력 방식", max_length=10, choices=InputType.choices)
    # 영상 파일 자체는 Media 폴더에, DB에는 경로만 저장
    video = models.FileField("영상", upload_to=upload_path, blank=True)
    # AI 서버가 그려 준 Skeleton Overlay 결과 영상 (H.264 MP4, 없으면 빈 값)
    overlay_video = models.FileField("Overlay 영상", upload_to="overlays/", blank=True)
    camera_name = models.CharField("카메라 이름", max_length=50, blank=True)
    capture_seconds = models.PositiveSmallIntegerField("수집 시간(초)", null=True, blank=True)
    # 사용자가 '하려던' 동작. 검증된 정답 Label이 아님
    expected_action = models.CharField(
        "수행하려던 동작", max_length=4, choices=ACTION_CHOICES, blank=True
    )

    status = models.CharField(
        "상태", max_length=10, choices=Status.choices, default=Status.PENDING, db_index=True
    )
    stage = models.CharField("진행 단계", max_length=50, blank=True)
    celery_task_id = models.CharField(max_length=50, blank=True)
    result_status = models.CharField("AI 응답 상태", max_length=20, blank=True)
    result_message = models.CharField("AI 응답 메시지", max_length=200, blank=True)
    error_message = models.TextField("오류 메시지", blank=True)
    video_duration_sec = models.FloatField("영상 길이(초)", null=True, blank=True)
    video_fps = models.FloatField(null=True, blank=True)
    processing_ms = models.PositiveIntegerField("AI 처리 시간(ms)", null=True, blank=True)

    created_at = models.DateTimeField("생성 시각", auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    # IP Webcam에서 첫 프레임을 받은 시각 → 화면의 실시간 녹화 초 카운터 기준
    capture_started_at = models.DateTimeField("녹화 시작 시각", null=True, blank=True)
    completed_at = models.DateTimeField("완료 시각", null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    @property
    def capture_elapsed(self):
        """녹화 중이면 시작 후 지난 초(서버 시계 기준), 아니면 None."""
        if self.capture_started_at is None or self.status != self.Status.STARTED:
            return None
        if not self.stage.startswith("영상 수집"):
            return None
        from django.utils import timezone

        return round((timezone.now() - self.capture_started_at).total_seconds(), 2)

    def __str__(self):
        return f"#{self.pk} {self.get_input_type_display()} {self.status}"

    @property
    def is_finished(self) -> bool:
        return self.status in (self.Status.SUCCESS, self.Status.FAILURE)

    @property
    def expected_action_name(self) -> str:
        return ACTION_NAMES.get(self.expected_action, "")

    @property
    def total_seconds(self):
        if self.completed_at and self.created_at:
            return round((self.completed_at - self.created_at).total_seconds(), 1)
        return None


class PersonPrediction(models.Model):
    """작업 1건 안에서 추적된 사람 1명의 예측 결과."""

    job = models.ForeignKey(AnalysisJob, on_delete=models.CASCADE, related_name="predictions")
    # 이 영상 안에서만 의미 있는 추적 번호 (영구적인 사람 ID가 아님)
    track_id = models.PositiveIntegerField()
    action_id = models.CharField(max_length=4, choices=ACTION_CHOICES, db_index=True)
    action_label = models.CharField(max_length=30)
    confidence = models.FloatField()
    is_uncertain = models.BooleanField(default=False)
    top3 = models.JSONField(default=list, blank=True)
    window_count = models.PositiveIntegerField(default=0)
    frame_count = models.PositiveIntegerField(default=0)
    duration_sec = models.FloatField(default=0)
    mean_keypoint_conf = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField("예측 시각", auto_now_add=True)

    class Meta:
        ordering = ["job", "track_id"]
        constraints = [
            models.UniqueConstraint(fields=["job", "track_id"], name="unique_track_per_job")
        ]

    def __str__(self):
        return f"job#{self.job_id} track{self.track_id} {self.action_label} {self.confidence:.2f}"

    @property
    def action_name(self) -> str:
        return ACTION_NAMES.get(self.action_id, self.action_id)
