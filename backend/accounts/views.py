from django.contrib import messages
from django.contrib.auth import login
from django.shortcuts import redirect, render

from .forms import SignUpForm


def signup(request):
    """회원가입 후 바로 로그인시키고 분석 화면으로 이동합니다."""
    if request.user.is_authenticated:
        return redirect("analyses:index")
    form = SignUpForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, f"{user.username}님, 가입을 환영합니다.")
        return redirect("analyses:index")
    return render(request, "accounts/signup.html", {"form": form})
