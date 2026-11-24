from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class SignUpForm(UserCreationForm):
    """Django 기본 User를 그대로 사용하는 회원가입 폼 (커스텀 회원 테이블 없음)."""

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email")
