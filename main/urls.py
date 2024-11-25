from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed', views.hand_gesture, name='video_feed'),
    path('handle_prompt/', views.handle_prompt, name='handle_prompt'),
    path('ready/', views.ready, name='ready'),
]