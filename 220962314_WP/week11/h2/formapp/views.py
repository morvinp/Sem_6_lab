from django.shortcuts import render, redirect
from .models import Product
from .forms import ProductForm

def product_entry(request):
    form = ProductForm()
    
    if request.method == "POST":
        form = ProductForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('index')  # Redirect to index page after saving

    return render(request, 'product_entry.html', {'form': form})

def index(request):
    products = Product.objects.all()  # Retrieve all products
    return render(request, 'index.html', {'products': products})
