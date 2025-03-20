from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Define the home URL
    path('add_work/', views.add_work_entry, name='add_work'),
    path('search_employees/', views.search_employees, name='search_employees'),
]
