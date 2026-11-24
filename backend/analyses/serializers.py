"""입력 검증과 JSON 응답 형태.

- AnalysisJobCreateSerializer : 생성(POST) 입력 검사
- AnalysisJobSerializer       : 조회 응답 + 수정(PATCH) — 제목·메모만 수정 가능
- AnalysisJobStatusSerializer : 상태 조회용 가벼운 응답
"""

from pathlib import Path

from django.conf import settings
from rest_framework import serializers

from .models import AnalysisJob, PersonPrediction

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class PersonPredictionSerializer(serializers.ModelSerializer):
    action_name = serializers.CharField(read_only=True)

    class Meta:
        model = PersonPrediction
        fields = [
            "track_id", "action_id", "action_label", "action_name", "confidence",
            "is_uncertain", "top3", "window_count", "frame_count", "duration_sec",
            "mean_keypoint_conf", "created_at",
        ]
        read_only_fields = fields


class AnalysisJobCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisJob
        fields = [
            "id", "title", "memo", "input_type", "video", "camera_name",
            "capture_seconds", "expected_action",
        ]
        read_only_fields = ["id"]

    def validate_video(self, video):
        if not video:
            return video
        suffix = Path(video.name).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise serializers.ValidationError(
                f"영상 파일만 올릴 수 있습니다. ({', '.join(sorted(VIDEO_EXTENSIONS))})"
            )
        if video.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise serializers.ValidationError(
                f"파일이 너무 큽니다. 최대 {settings.MAX_UPLOAD_MB}MB"
            )
        return video

    def validate(self, attrs):
        input_type = attrs.get("input_type")
        if input_type == AnalysisJob.InputType.UPLOAD:
            if not attrs.get("video"):
                raise serializers.ValidationError({"video": "분석할 영상을 선택하세요."})
            attrs["camera_name"] = ""
            attrs["capture_seconds"] = None
        elif input_type == AnalysisJob.InputType.IP_WEBCAM:
            if attrs.get("video"):
                raise serializers.ValidationError(
                    {"video": "IP Webcam 분석에는 영상을 첨부하지 않습니다."}
                )
            camera = attrs.get("camera_name", "")
            # 허용 목록에 있는 카메라 '이름'만 받습니다. URL은 서버 설정에만 있음
            if camera not in settings.ALLOWED_CAMERAS:
                raise serializers.ValidationError(
                    {"camera_name": "허용된 카메라가 아닙니다."}
                )
            seconds = attrs.get("capture_seconds") or settings.CAMERA_CAPTURE_SECONDS
            if not settings.CAMERA_MIN_SECONDS <= seconds <= settings.CAMERA_MAX_SECONDS:
                raise serializers.ValidationError(
                    {
                        "capture_seconds": (
                            f"수집 시간은 {settings.CAMERA_MIN_SECONDS}~"
                            f"{settings.CAMERA_MAX_SECONDS}초입니다."
                        )
                    }
                )
            attrs["capture_seconds"] = seconds
        return attrs


class AnalysisJobSerializer(serializers.ModelSerializer):
    """조회 응답. 수정 가능한 필드는 title, memo 뿐입니다."""

    username = serializers.CharField(source="user.username", read_only=True)
    input_type_display = serializers.CharField(source="get_input_type_display", read_only=True)
    status_display = serializers.CharField(source="get_status_display", read_only=True)
    expected_action_name = serializers.CharField(read_only=True)
    is_finished = serializers.BooleanField(read_only=True)
    total_seconds = serializers.FloatField(read_only=True)
    video_url = serializers.SerializerMethodField()
    overlay_url = serializers.SerializerMethodField()
    predictions = PersonPredictionSerializer(many=True, read_only=True)

    class Meta:
        model = AnalysisJob
        fields = [
            "id", "username", "title", "memo", "input_type", "input_type_display",
            "camera_name", "capture_seconds", "expected_action", "expected_action_name",
            "status", "status_display", "stage", "is_finished", "celery_task_id",
            "result_status", "result_message", "error_message",
            "video_url", "overlay_url", "video_duration_sec", "video_fps", "processing_ms", "total_seconds",
            "created_at", "started_at", "completed_at", "predictions",
        ]
        read_only_fields = [f for f in fields if f not in ("title", "memo")]

    def get_video_url(self, job):
        return job.video.url if job.video else None

    def get_overlay_url(self, job):
        return job.overlay_video.url if job.overlay_video else None


class AnalysisJobStatusSerializer(serializers.ModelSerializer):
    status_display = serializers.CharField(source="get_status_display", read_only=True)
    is_finished = serializers.BooleanField(read_only=True)
    prediction_count = serializers.IntegerField(source="predictions.count", read_only=True)

    class Meta:
        model = AnalysisJob
        fields = [
            "id", "status", "status_display", "stage", "is_finished",
            "error_message", "result_message", "prediction_count", "completed_at",
        ]
        read_only_fields = fields
