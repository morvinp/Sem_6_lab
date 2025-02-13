from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Using path() instead of url()
]
