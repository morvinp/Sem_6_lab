from django.urls import path
from . import views

urlpatterns = [
    path('calculate/', views.calculate, name='calculate'),
    path('result/', views.result, name='result'),
]
