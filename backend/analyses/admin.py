from django.contrib import admin

from .models import AnalysisJob, PersonPrediction


class PersonPredictionInline(admin.TabularInline):
    model = PersonPrediction
    extra = 0
    can_delete = False
    fields = ("track_id", "action_id", "action_label", "confidence", "is_uncertain",
              "window_count", "duration_sec", "mean_keypoint_conf")
    readonly_fields = fields


@admin.register(AnalysisJob)
class AnalysisJobAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "input_type", "status", "title", "expected_action",
                    "prediction_count", "processing_ms", "created_at")
    list_filter = ("status", "input_type", "expected_action", "created_at")
    search_fields = ("user__username", "title", "memo", "camera_name")
    readonly_fields = ("celery_task_id", "created_at", "started_at", "completed_at",
                       "processing_ms", "video_duration_sec", "video_fps",
                       "result_status", "result_message")
    inlines = [PersonPredictionInline]
    date_hierarchy = "created_at"

    @admin.display(description="예측 수")
    def prediction_count(self, obj):
        return obj.predictions.count()


@admin.register(PersonPrediction)
class PersonPredictionAdmin(admin.ModelAdmin):
    list_display = ("id", "job", "job_user", "track_id", "action_id", "action_label",
                    "confidence", "is_uncertain", "created_at")
    list_filter = ("action_id", "is_uncertain", "job__input_type", "job__status")
    search_fields = ("job__user__username", "action_label", "job__id")
    list_select_related = ("job__user",)

    @admin.display(description="사용자", ordering="job__user__username")
    def job_user(self, obj):
        return obj.job.user.username
