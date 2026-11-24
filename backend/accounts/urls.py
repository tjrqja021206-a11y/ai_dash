from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

urlpatterns = [
    path("signup/", views.signup, name="signup"),
    path("login/", auth_views.LoginView.as_view(), name="login"),
    # Django 5+는 로그아웃을 POST로만 허용 (base.html의 form 사용)
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
]
