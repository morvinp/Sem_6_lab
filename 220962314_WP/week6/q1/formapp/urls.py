from django.urls import path
from . import views

urlpatterns = [
    path('', views.calculate, name='calculate'),  # Using path() instead of url()
]
