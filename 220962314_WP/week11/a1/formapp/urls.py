from django.urls import path
from .views import student_view

urlpatterns = [
    path('', student_view, name='student_view'),
]
