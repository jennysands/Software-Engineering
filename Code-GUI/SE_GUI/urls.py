from django.urls import path
from . import views

urlpatterns = [
    
    path("", views.index, name="index"),
    path("image_predict/", views.image_predict, name="image_predict"),
    path("audio_predict/", views.audio_predict, name="audio_predict"),
]