from django.contrib import admin
from .models import Product

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ('title', 'price', 'description')  # Fields to display in admin list
    search_fields = ('title', 'description')  # Enable search functionality
    list_filter = ('price',)  # Filter by price
