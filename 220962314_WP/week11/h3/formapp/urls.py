from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('get-human/<int:human_id>/', views.get_human_details, name='get_human_details'),
    path('update-human/<int:human_id>/', views.update_human, name='update_human'),
    path('delete-human/<int:human_id>/', views.delete_human, name='delete_human'),
]
