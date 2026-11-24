"""Celery 앱: Django 설정(CELERY_*)을 읽고 각 앱의 tasks.py를 자동으로 찾습니다.

Worker 실행 (backend 폴더에서):
    celery -A config worker -l info
"""

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
