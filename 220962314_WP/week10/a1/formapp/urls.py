from django.urls import path
from .views import institute_list, institute_details

urlpatterns = [
    path('institutes/', institute_list, name='institute_list'),
    path('institutes/<int:institute_id>/', institute_details, name='institute_details'),
]
