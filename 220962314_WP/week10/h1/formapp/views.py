from django.shortcuts import render, redirect
from .models import Category, Page
from .forms import CategoryForm, PageForm

def index(request):
    categories = Category.objects.prefetch_related('page_set').all()
    return render(request, 'index.html', {'categories': categories})    
def add_category(request):
    if request.method == "POST":
        form = CategoryForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('index')
    else:
        form = CategoryForm()
    return render(request, 'add_category.html', {'form': form})


def add_page(request):
    if request.method == 'POST':
        form = PageForm(request.POST)
        if form.is_valid():
            page = form.save()
            print(f"Page added: {page.title}, URL: {page.url}, Category: {page.category}")  # Debugging print
            return redirect('index')
        else:
            print("Form is invalid:", form.errors)  # Debugging print
    else:
        form = PageForm()
    
    return render(request, 'add_page.html', {'form': form})