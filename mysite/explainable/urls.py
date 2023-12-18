from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("bias", views.bias, name = "bias"),
    path("performance", views.performance, name = "performance"),
    path("enviromental", views.enviromental, name = "enviromental"),
    path("ethical", views.ethical, name = "ethical")

]

