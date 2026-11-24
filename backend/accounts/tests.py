from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse


class AccountTests(TestCase):
    def test_signup_creates_user_and_logs_in(self):
        response = self.client.post(
            reverse("signup"),
            {
                "username": "alice",
                "email": "alice@example.com",
                "password1": "S3cure-pass-123",
                "password2": "S3cure-pass-123",
            },
        )
        self.assertRedirects(response, reverse("analyses:index"))
        self.assertTrue(User.objects.filter(username="alice").exists())

    def test_duplicate_username_rejected(self):
        User.objects.create_user("bob", password="S3cure-pass-123")
        response = self.client.post(
            reverse("signup"),
            {"username": "bob", "password1": "S3cure-pass-123", "password2": "S3cure-pass-123"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(User.objects.filter(username="bob").count(), 1)

    def test_password_mismatch_rejected(self):
        response = self.client.post(
            reverse("signup"),
            {"username": "carol", "password1": "S3cure-pass-123", "password2": "different-456"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username="carol").exists())

    def test_login_and_logout(self):
        User.objects.create_user("dave", password="S3cure-pass-123")
        response = self.client.post(
            reverse("login"), {"username": "dave", "password": "S3cure-pass-123"}
        )
        self.assertRedirects(response, reverse("analyses:index"))
        response = self.client.post(reverse("logout"))
        self.assertRedirects(response, reverse("login"))

    def test_anonymous_redirected_from_analysis_page(self):
        response = self.client.get(reverse("analyses:index"))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("login"), response["Location"])
