from django.contrib import admin
from .models import Institute

@admin.register(Institute)
class InstituteAdmin(admin.ModelAdmin):
    list_display = ('institute_id', 'name', 'no_of_courses')
    search_fields = ('name',)
    list_filter = ('no_of_courses',)
