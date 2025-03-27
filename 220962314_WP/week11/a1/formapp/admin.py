from django.contrib import admin
from .models import Student

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ('student_id', 'student_name', 'course_name', 'date_of_birth')  # Columns in admin panel
    search_fields = ('student_id', 'student_name', 'course_name')  # Search by these fields
    list_filter = ('course_name',)  # Filter students by course
