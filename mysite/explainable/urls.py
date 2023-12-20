from django.urls import path

from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path("", views.index, name="index"),
    path("bias", views.bias, name = "bias"),
    path("performance", views.performance, name = "performance"),
    path("enviromental", views.enviromental, name = "enviromental"),
    path("ethical", views.ethical, name = "ethical")
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

