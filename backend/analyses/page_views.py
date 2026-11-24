"""HTML 화면: 사용자 분석 화면과 관리자 전용 대시보드."""

from django.conf import settings
from django.contrib.auth.decorators import login_required, user_passes_test
from django.db.models import Avg, Count
from django.shortcuts import render

from .actions import ACTIONS
from .models import AnalysisJob, PersonPrediction


@login_required
def index(request):
    return render(
        request,
        "analyses/index.html",
        {
            "actions": ACTIONS,
            "cameras": list(settings.ALLOWED_CAMERAS),
            "capture_seconds": settings.CAMERA_CAPTURE_SECONDS,
            "min_seconds": settings.CAMERA_MIN_SECONDS,
            "max_seconds": settings.CAMERA_MAX_SECONDS,
            "uncertain_threshold": settings.UNCERTAIN_THRESHOLD,
        },
    )


def _is_staff(user):
    return user.is_active and user.is_staff


@login_required
@user_passes_test(_is_staff, login_url="login")
def dashboard(request):
    """모든 숫자는 PostgreSQL의 실제 데이터를 Django ORM으로 집계한 값입니다."""
    action_counts = dict(
        PersonPrediction.objects.values_list("action_id").annotate(n=Count("id"))
    )
    status_counts = dict(AnalysisJob.objects.values_list("status").annotate(n=Count("id")))
    input_counts = dict(AnalysisJob.objects.values_list("input_type").annotate(n=Count("id")))
    avg_ms = AnalysisJob.objects.filter(status=AnalysisJob.Status.SUCCESS).aggregate(
        avg=Avg("processing_ms")
    )["avg"]

    chart_data = {
        "actions": {
            "labels": [f"{aid} {name}" for aid, _, name in ACTIONS],
            "values": [action_counts.get(aid, 0) for aid, _, _ in ACTIONS],
        },
        "status": {
            "labels": [label for _, label in AnalysisJob.Status.choices],
            "keys": [key for key, _ in AnalysisJob.Status.choices],
            "values": [status_counts.get(key, 0) for key, _ in AnalysisJob.Status.choices],
        },
        "input": {
            "labels": [label for _, label in AnalysisJob.InputType.choices],
            "values": [input_counts.get(key, 0) for key, _ in AnalysisJob.InputType.choices],
        },
    }
    context = {
        "chart_data": chart_data,
        "total_jobs": sum(status_counts.values()),
        "total_predictions": sum(action_counts.values()),
        "success_jobs": status_counts.get(AnalysisJob.Status.SUCCESS, 0),
        "failure_jobs": status_counts.get(AnalysisJob.Status.FAILURE, 0),
        "avg_seconds": round(avg_ms / 1000, 1) if avg_ms else None,
        "recent_jobs": AnalysisJob.objects.select_related("user")
        .annotate(n_pred=Count("predictions"))[:15],
    }
    return render(request, "analyses/dashboard.html", context)
