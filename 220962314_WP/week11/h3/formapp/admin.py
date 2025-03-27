from django.contrib import admin
from .models import Human

@admin.register(Human)
class HumanAdmin(admin.ModelAdmin):
    list_display = ('first_name', 'last_name', 'phone', 'city')
    search_fields = ('first_name', 'last_name', 'city')
