"""DRF 분석 작업 API.

POST   /api/v1/analyses/              작업 생성 → DB 저장 완료 후 Celery 등록 → 작업 ID 즉시 반환
GET    /api/v1/analyses/              내 작업 목록 (관리자는 전체)
GET    /api/v1/analyses/{id}/         상태 + 사람별 결과
PATCH  /api/v1/analyses/{id}/         제목·메모만 수정
DELETE /api/v1/analyses/{id}/         작업·사람별 결과·영상 파일 삭제 (처리 중이면 거부)
GET    /api/v1/analyses/{id}/status/  상태만 가볍게 조회 (브라우저 폴링용)
GET    /api/v1/analyses/cameras/      선택 가능한 카메라 이름 목록
"""

from django.conf import settings
from django.db import transaction
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import AnalysisJob
from .serializers import (
    AnalysisJobCreateSerializer,
    AnalysisJobSerializer,
    AnalysisJobStatusSerializer,
)
from .tasks import run_analysis


class AnalysisJobViewSet(viewsets.ModelViewSet):
    http_method_names = ["get", "post", "patch", "delete", "head", "options"]

    def get_queryset(self):
        queryset = AnalysisJob.objects.select_related("user").prefetch_related("predictions")
        if self.request.user.is_staff:
            return queryset
        # 일반 회원은 본인 작업만 보입니다. 남의 작업 id로 요청하면 404
        return queryset.filter(user=self.request.user)

    def get_serializer_class(self):
        if self.action == "create":
            return AnalysisJobCreateSerializer
        if self.action == "status":
            return AnalysisJobStatusSerializer
        return AnalysisJobSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        job = serializer.save(user=request.user, stage="대기열 등록")

        # DB 트랜잭션이 커밋된 뒤에 작업을 큐에 넣어야
        # Worker가 아직 저장되지 않은 Job을 찾지 못하는 문제가 생기지 않습니다.
        def enqueue():
            result = run_analysis.delay(job.pk)
            AnalysisJob.objects.filter(pk=job.pk).update(celery_task_id=result.id or "")

        transaction.on_commit(enqueue)

        return Response(
            {"id": job.pk, "status": job.status, "status_url": f"/api/v1/analyses/{job.pk}/status/"},
            status=status.HTTP_202_ACCEPTED,
        )

    def perform_destroy(self, instance):
        if instance.status == AnalysisJob.Status.STARTED:
            from rest_framework.exceptions import ValidationError

            raise ValidationError("처리 중인 작업은 삭제할 수 없습니다. 완료 후 삭제하세요.")
        video = instance.video
        instance.delete()  # PersonPrediction은 CASCADE로 함께 삭제
        if video:
            video.delete(save=False)

    @action(detail=True, methods=["get"])
    def status(self, request, pk=None):
        return Response(self.get_serializer(self.get_object()).data)

    @action(detail=False, methods=["get"])
    def cameras(self, request):
        return Response(
            {
                "cameras": list(settings.ALLOWED_CAMERAS),
                "default_seconds": settings.CAMERA_CAPTURE_SECONDS,
                "min_seconds": settings.CAMERA_MIN_SECONDS,
                "max_seconds": settings.CAMERA_MAX_SECONDS,
            }
        )
