from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import page_views
from .views import AnalysisJobViewSet

app_name = "analyses"

router = DefaultRouter()
router.register("analyses", AnalysisJobViewSet, basename="analysis")

urlpatterns = [
    path("analyses/", page_views.index, name="index"),
    path("dashboard/", page_views.dashboard, name="dashboard"),
    path("api/v1/", include(router.urls)),
]
