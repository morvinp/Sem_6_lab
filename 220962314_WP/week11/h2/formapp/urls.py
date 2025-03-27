from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('add-product/', views.product_entry, name='product_entry'),
]
