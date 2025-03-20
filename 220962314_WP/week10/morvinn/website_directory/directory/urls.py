# directory/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.directory_view, name='directory_view'),
    path('add_category/', views.add_category, name='add_category'),
    path('add_page/', views.add_page, name='add_page'),
]
