"""분석 API 권한·상태·결과 저장 테스트.

FastAPI·카메라는 mock으로 대체하고 CELERY_TASK_ALWAYS_EAGER로 작업을 즉시 실행합니다.
실행: python manage.py test
"""

import shutil
import tempfile
from unittest import mock

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from .models import AnalysisJob, PersonPrediction
from .services import AIServiceError

TEMP_MEDIA = tempfile.mkdtemp()

FAKE_RESULT = {
    "status": "ok",
    "message": "2명의 동작을 예측했습니다.",
    "video": {"fps": 30.0, "duration_sec": 4.0},
    "processing_ms": 1234,
    "predictions": [
        {"track_id": 1, "action_id": "A003", "action_label": "front_kick", "confidence": 0.81,
         "top3": [], "window_count": 9, "frame_count": 120, "duration_sec": 4.0,
         "mean_keypoint_conf": 0.9},
        {"track_id": 2, "action_id": "A007", "action_label": "sit_floor", "confidence": 0.31,
         "top3": [], "window_count": 9, "frame_count": 110, "duration_sec": 3.7,
         "mean_keypoint_conf": 0.8},
    ],
}


def fake_video(name="clip.mp4"):
    return SimpleUploadedFile(name, b"\x00" * 1024, content_type="video/mp4")


@override_settings(
    MEDIA_ROOT=TEMP_MEDIA,
    CELERY_TASK_ALWAYS_EAGER=True,
    ALLOWED_CAMERAS={"phone": "http://192.168.0.10:8080/video"},
    CAMERA_ROTATION={"phone": 90},
)
class AnalysisApiTests(TestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEMP_MEDIA, ignore_errors=True)

    def setUp(self):
        self.alice = User.objects.create_user("alice", password="pw-alice-123")
        self.bob = User.objects.create_user("bob", password="pw-bob-123")
        self.admin = User.objects.create_user("admin", password="pw-admin-123", is_staff=True)
        self.client = APIClient()

    def create_upload(self, user):
        self.client.force_authenticate(user)
        with self.captureOnCommitCallbacks(execute=True):
            response = self.client.post(
                "/api/v1/analyses/",
                {"input_type": "UPLOAD", "video": fake_video(), "expected_action": "A003"},
                format="multipart",
            )
        return response

    # T05 비동기: 작업 ID를 즉시 반환
    @mock.patch("analyses.tasks.predict_video", return_value=FAKE_RESULT)
    def test_create_returns_job_id_and_saves_predictions(self, _):
        response = self.create_upload(self.alice)
        self.assertEqual(response.status_code, 202)
        job = AnalysisJob.objects.get(pk=response.data["id"])
        # T04·T06: 한 작업에 Track ID별 결과 2건 저장
        self.assertEqual(job.status, AnalysisJob.Status.SUCCESS)
        self.assertEqual(job.predictions.count(), 2)
        self.assertTrue(job.predictions.get(track_id=2).is_uncertain)
        self.assertEqual(job.processing_ms, 1234)

    def test_anonymous_rejected(self):
        response = self.client.get("/api/v1/analyses/")
        self.assertIn(response.status_code, (401, 403))

    def test_invalid_input_rejected(self):
        self.client.force_authenticate(self.alice)
        no_video = self.client.post("/api/v1/analyses/", {"input_type": "UPLOAD"}, format="multipart")
        wrong_ext = self.client.post(
            "/api/v1/analyses/",
            {"input_type": "UPLOAD", "video": fake_video("notes.txt")},
            format="multipart",
        )
        bad_camera = self.client.post(
            "/api/v1/analyses/",
            {"input_type": "IP_WEBCAM", "camera_name": "http://evil.example/"},
            format="multipart",
        )
        too_long = self.client.post(
            "/api/v1/analyses/",
            {"input_type": "IP_WEBCAM", "camera_name": "phone", "capture_seconds": 60},
            format="multipart",
        )
        for response in (no_video, wrong_ext, bad_camera, too_long):
            self.assertEqual(response.status_code, 400, response.data)
        self.assertEqual(AnalysisJob.objects.count(), 0)

    # T07 CRUD 권한
    @mock.patch("analyses.tasks.predict_video", return_value=FAKE_RESULT)
    def test_other_user_cannot_read_update_delete(self, _):
        job_id = self.create_upload(self.alice).data["id"]
        self.client.force_authenticate(self.bob)
        self.assertEqual(self.client.get(f"/api/v1/analyses/{job_id}/").status_code, 404)
        self.assertEqual(
            self.client.patch(f"/api/v1/analyses/{job_id}/", {"memo": "x"}).status_code, 404
        )
        self.assertEqual(self.client.delete(f"/api/v1/analyses/{job_id}/").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/analyses/").data["count"], 0)
        # 관리자는 전체 조회
        self.client.force_authenticate(self.admin)
        self.assertEqual(self.client.get(f"/api/v1/analyses/{job_id}/").status_code, 200)

    @mock.patch("analyses.tasks.predict_video", return_value=FAKE_RESULT)
    def test_patch_only_title_and_memo(self, _):
        job_id = self.create_upload(self.alice).data["id"]
        response = self.client.patch(
            f"/api/v1/analyses/{job_id}/",
            {"title": "앞차기 연습", "memo": "1차", "status": "FAILURE"},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        job = AnalysisJob.objects.get(pk=job_id)
        self.assertEqual(job.title, "앞차기 연습")
        self.assertEqual(job.status, AnalysisJob.Status.SUCCESS)  # 상태는 바뀌지 않음

    @mock.patch("analyses.tasks.predict_video", return_value=FAKE_RESULT)
    def test_delete_removes_predictions(self, _):
        job_id = self.create_upload(self.alice).data["id"]
        self.assertEqual(self.client.delete(f"/api/v1/analyses/{job_id}/").status_code, 204)
        self.assertFalse(PersonPrediction.objects.filter(job_id=job_id).exists())

    # Skeleton Overlay: AI 서버가 만든 영상을 받아 저장하고, 삭제 시 함께 지움
    @mock.patch("analyses.tasks.download_overlay", return_value=b"fake-mp4")
    @mock.patch(
        "analyses.tasks.predict_video",
        return_value={**FAKE_RESULT, "overlay": {"url": "/overlay/" + "a" * 32}},
    )
    def test_overlay_saved_and_deleted_with_job(self, _, download):
        job_id = self.create_upload(self.alice).data["id"]
        download.assert_called_once_with("/overlay/" + "a" * 32)
        job = AnalysisJob.objects.get(pk=job_id)
        self.assertEqual(job.status, AnalysisJob.Status.SUCCESS)
        overlay_path = job.overlay_video.path
        with open(overlay_path, "rb") as file:
            self.assertEqual(file.read(), b"fake-mp4")
        detail = self.client.get(f"/api/v1/analyses/{job_id}/").data
        self.assertTrue(detail["overlay_url"].endswith(f"overlay_job_{job_id}.mp4"))

        self.assertEqual(self.client.delete(f"/api/v1/analyses/{job_id}/").status_code, 204)
        self.assertFalse(__import__("os").path.exists(overlay_path))

    # Overlay를 못 받아도 분석 결과는 SUCCESS로 유지
    @mock.patch("analyses.tasks.download_overlay", side_effect=AIServiceError("HTTP 404"))
    @mock.patch(
        "analyses.tasks.predict_video",
        return_value={**FAKE_RESULT, "overlay": {"url": "/overlay/" + "b" * 32}},
    )
    def test_overlay_failure_keeps_success(self, *_):
        job = AnalysisJob.objects.get(pk=self.create_upload(self.alice).data["id"])
        self.assertEqual(job.status, AnalysisJob.Status.SUCCESS)
        self.assertEqual(job.predictions.count(), 2)
        self.assertFalse(job.overlay_video)

    # T09 예외: AI 서버 장애 → FAILURE와 오류 기록
    @mock.patch("analyses.tasks.predict_video", side_effect=AIServiceError("AI 서버에 연결할 수 없습니다."))
    def test_ai_failure_recorded(self, _):
        job_id = self.create_upload(self.alice).data["id"]
        job = AnalysisJob.objects.get(pk=job_id)
        self.assertEqual(job.status, AnalysisJob.Status.FAILURE)
        self.assertIn("연결할 수 없습니다", job.error_message)
        status = self.client.get(f"/api/v1/analyses/{job_id}/status/").data
        self.assertTrue(status["is_finished"])

    # T09 예외: 카메라 연결 실패
    @mock.patch("analyses.tasks.capture_clip")
    def test_camera_failure_recorded(self, capture):
        from .camera_capture import CameraError

        capture.side_effect = CameraError("카메라에 연결할 수 없습니다.")
        self.client.force_authenticate(self.alice)
        with self.captureOnCommitCallbacks(execute=True):
            response = self.client.post(
                "/api/v1/analyses/",
                {"input_type": "IP_WEBCAM", "camera_name": "phone", "capture_seconds": 8},
                format="multipart",
            )
        job = AnalysisJob.objects.get(pk=response.data["id"])
        self.assertEqual(job.status, AnalysisJob.Status.FAILURE)
        self.assertIn("카메라", job.error_message)
        capture.assert_called_once()
        self.assertEqual(capture.call_args.args[0], "http://192.168.0.10:8080/video")
        self.assertEqual(capture.call_args.kwargs["rotate"], 90)  # CAMERA_ROTATION 전달


def _jpeg(width=64, height=32):
    import cv2
    import numpy as np

    return cv2.imencode(".jpg", np.zeros((height, width, 3), np.uint8))[1].tobytes()


@override_settings(
    ALLOWED_CAMERAS={"phone": "http://192.168.0.10:8080/video"},
    CAMERA_ROTATION={"phone": 90},
)
class CameraPreviewTests(TestCase):
    """실시간 미리보기: 허용 카메라만, 회전 적용, AI 서버 판정 헤더 전달."""

    URL = "/api/v1/analyses/cameras/{}/preview/"

    def setUp(self):
        self.client = APIClient()
        self.client.force_authenticate(User.objects.create_user("cam", password="pw-cam-123"))

    def test_login_required(self):
        self.assertEqual(APIClient().get(self.URL.format("phone")).status_code, 403)

    def test_unknown_camera_rejected(self):
        self.assertEqual(self.client.get(self.URL.format("evil")).status_code, 404)

    @mock.patch("analyses.camera_preview.requests.post")
    @mock.patch("analyses.camera_preview.requests.get")
    def test_returns_annotated_jpeg_with_pose_status(self, get, post):
        get.return_value = mock.Mock(status_code=200, content=_jpeg(), raise_for_status=lambda: None)
        post.return_value = mock.Mock(
            status_code=200, content=b"annotated",
            headers={"X-Pose-Status": "legs_missing", "X-Pose-Keypoint-Conf": "0.71"},
        )
        response = self.client.get(self.URL.format("phone"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "image/jpeg")
        self.assertEqual(response.content, b"annotated")
        self.assertEqual(response["X-Pose-Status"], "legs_missing")
        get.assert_called_once()
        self.assertEqual(get.call_args.args[0], "http://192.168.0.10:8080/shot.jpg")

    @mock.patch("analyses.camera_preview.requests.post")
    @mock.patch("analyses.camera_preview.requests.get")
    def test_rotates_frame_and_works_without_ai(self, get, post):
        import cv2
        import numpy as np
        import requests

        get.return_value = mock.Mock(status_code=200, content=_jpeg(64, 32), raise_for_status=lambda: None)
        post.side_effect = requests.ConnectionError()
        response = self.client.get(self.URL.format("phone"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["X-Pose-Status"], "unknown")
        frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(frame.shape[:2], (64, 32))  # 가로 64×세로 32 → 90° 회전 후 세로 64

    @mock.patch("analyses.camera_preview.requests.get")
    def test_camera_down_returns_502(self, get):
        import requests

        get.side_effect = requests.ConnectionError()
        response = self.client.get(self.URL.format("phone"))
        self.assertEqual(response.status_code, 502)
        self.assertIn("카메라", response.json()["detail"])


@override_settings(FASTAPI_RETRIES=2, FASTAPI_RETRY_DELAY_SECONDS=0)
class AIServiceRetryTests(TestCase):
    """AI 서버 재시작·모델 로딩 중에는 재시도하고, 잘못된 영상(4xx)은 바로 실패."""

    def setUp(self):
        temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp.write(b"fake")
        temp.close()
        self.path = temp.name
        self.addCleanup(lambda: __import__("os").remove(self.path))

    @staticmethod
    def response(status, body):
        fake = mock.Mock(status_code=status)
        fake.json.return_value = body
        fake.text = str(body)
        return fake

    @mock.patch("analyses.services._post_video")
    def test_retries_connection_error_then_succeeds(self, post):
        import requests

        from .services import predict_video

        post.side_effect = [requests.ConnectionError(), self.response(200, FAKE_RESULT)]
        self.assertEqual(predict_video(self.path)["status"], "ok")
        self.assertEqual(post.call_count, 2)

    @mock.patch("analyses.services._post_video")
    def test_retries_503_then_gives_up(self, post):
        from .services import predict_video

        post.return_value = self.response(503, {"detail": "모델이 아직 준비되지 않았습니다."})
        with self.assertRaisesMessage(AIServiceError, "503"):
            predict_video(self.path)
        self.assertEqual(post.call_count, 3)  # 첫 시도 + 재시도 2번

    @mock.patch("analyses.services._post_video")
    def test_bad_video_not_retried(self, post):
        from .services import predict_video

        post.return_value = self.response(400, {"detail": "영상 파일을 열 수 없습니다."})
        with self.assertRaisesMessage(AIServiceError, "영상 파일을 열 수 없습니다"):
            predict_video(self.path)
        self.assertEqual(post.call_count, 1)


class DashboardAccessTests(TestCase):
    # T08 관리자 화면
    def test_dashboard_staff_only(self):
        User.objects.create_user("user", password="pw-user-123")
        User.objects.create_user("boss", password="pw-boss-123", is_staff=True)
        self.client.login(username="user", password="pw-user-123")
        self.assertEqual(self.client.get("/dashboard/").status_code, 302)
        self.client.login(username="boss", password="pw-boss-123")
        response = self.client.get("/dashboard/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("chart_data", response.context)
