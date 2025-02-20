from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home page
    path('metadata/', views.metadata, name='metadata'),
    path('reviews/', views.reviews, name='reviews'),
    path('publisher-info/', views.publisher_info, name='publisher_info'),
]
