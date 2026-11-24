"""Django 설정. 실제 값은 환경변수(.env → compose.yaml)로 주입합니다."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS", "http://localhost:8000")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "accounts",
    "analyses",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    # gunicorn은 정적 파일을 직접 주지 않으므로 WhiteNoise가 /static/을 제공
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# ----------------------------------------------------
# DB: 기본은 PostgreSQL. 로컬 빠른 테스트에서만 USE_SQLITE=1
# ----------------------------------------------------
if env_bool("USE_SQLITE"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("POSTGRES_DB", "vision_action"),
            "USER": os.getenv("POSTGRES_USER", "vision"),
            "PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
            "HOST": os.getenv("POSTGRES_HOST", "localhost"),
            "PORT": os.getenv("POSTGRES_PORT", "5432"),
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "ko-kr"
TIME_ZONE = "Asia/Seoul"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
WHITENOISE_USE_FINDERS = True  # collectstatic 없이 static/·Admin 정적 파일을 바로 제공

# 업로드 영상·카메라 수집 영상 저장 위치 (Django와 Celery가 같은 볼륨을 공유)
MEDIA_URL = "media/"
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", BASE_DIR / "media"))

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ----------------------------------------------------
# 로그인
# ----------------------------------------------------
LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "analyses:index"
LOGOUT_REDIRECT_URL = "login"

# ----------------------------------------------------
# DRF: 세션 로그인 사용자만 API 사용
# ----------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",  # curl 테스트용
    ],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
}

# ----------------------------------------------------
# Redis·Celery
# ----------------------------------------------------
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
CELERY_TASK_ALWAYS_EAGER = env_bool("CELERY_TASK_ALWAYS_EAGER")  # Redis 없이 테스트할 때만
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_ACKS_LATE = True
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_TIMEZONE = TIME_ZONE

# ----------------------------------------------------
# AI 서버·입력 제한
# ----------------------------------------------------
# 컨테이너끼리는 localhost가 아니라 Compose 서비스 이름(fastapi)으로 통신합니다.
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8001")
FASTAPI_TIMEOUT_SECONDS = int(os.getenv("FASTAPI_TIMEOUT_SECONDS", "300"))
# AI 서버 재시작·모델 로딩 중(연결 실패·503)일 때 재시도 횟수와 간격(초, 회차마다 늘어남)
FASTAPI_RETRIES = int(os.getenv("FASTAPI_RETRIES", "3"))
# 결과 화면용 Skeleton Overlay 영상을 함께 만들지 (영상 1번 더 읽음: 3.7초 60fps 영상 기준 +1.8초)
ANALYSIS_OVERLAY = env_bool("ANALYSIS_OVERLAY", True)
FASTAPI_RETRY_DELAY_SECONDS = float(os.getenv("FASTAPI_RETRY_DELAY_SECONDS", "3"))
# Celery 작업 하나의 최대 시간: 카메라 수집 + AI 응답 대기 + 여유. 넘으면 FAILURE로 기록
ANALYSIS_TIME_LIMIT_SECONDS = int(
    os.getenv("ANALYSIS_TIME_LIMIT_SECONDS", str(FASTAPI_TIMEOUT_SECONDS + 120))
)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024


def parse_cameras(raw: str) -> dict[str, str]:
    """"이름=URL,이름=URL" → {이름: URL}. 사용자는 이름만 고를 수 있습니다."""
    cameras = {}
    for item in raw.split(","):
        if "=" in item:
            name, url = item.split("=", 1)
            cameras[name.strip()] = url.strip()
    return cameras


# 허용된 IP Webcam 목록 (사용자가 임의 URL을 입력하지 못하게 함)
ALLOWED_CAMERAS = parse_cameras(os.getenv("ALLOWED_CAMERAS", ""))
# 카메라별 시계 방향 회전(0/90/180/270). 휴대폰을 세로로 들었는데 앱이 가로로 보내면 사람이
# 옆으로 누운 영상이 되어 '눕기' 계열로 오인하므로, 수집·미리보기 단계에서 바로 세웁니다.
# 예: CAMERA_ROTATION=phone1=90
CAMERA_ROTATION = {
    name: int(value) % 360
    for name, value in parse_cameras(os.getenv("CAMERA_ROTATION", "")).items()
    if value.strip().lstrip("-").isdigit() and int(value) % 90 == 0
}
# 수집 영상의 긴 변 최대 픽셀. YOLO는 640으로 줄여 쓰고 Feature는 몸통 길이로 정규화되므로
# 줄여도 결과는 같고, 1080p 원본을 그대로 모을 때의 메모리(8초에 약 3GB)를 크게 줄입니다.
CAMERA_MAX_SIDE = int(os.getenv("CAMERA_MAX_SIDE", "1280"))
CAMERA_CAPTURE_SECONDS = int(os.getenv("CAMERA_CAPTURE_SECONDS", "8"))
CAMERA_MIN_SECONDS = 5
CAMERA_MAX_SECONDS = 10
CAMERA_OPEN_TIMEOUT_SECONDS = int(os.getenv("CAMERA_OPEN_TIMEOUT_SECONDS", "10"))
# 이 값 미만이면 화면에 '판단 불확실' 표시. AI 서버가 is_uncertain을 보내면 그 값을 쓰고, 이 값은
# 화면 안내와 예비 판단용 (project.json calibration.uncertain_threshold와 같게: 정답 가능성 0.8)
UNCERTAIN_THRESHOLD = float(os.getenv("UNCERTAIN_THRESHOLD", "0.8"))

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "loggers": {"analyses": {"handlers": ["console"], "level": "INFO"}},
}
