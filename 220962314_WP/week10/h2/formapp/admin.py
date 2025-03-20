from django.contrib import admin
from .models import Works, Lives

@admin.register(Works)
class WorksAdmin(admin.ModelAdmin):
    list_display = ('person_name', 'company_name', 'salary')
    search_fields = ('person_name', 'company_name')

@admin.register(Lives)
class LivesAdmin(admin.ModelAdmin):
    list_display = ('person_name', 'street', 'city')
    search_fields = ('person_name', 'city')
