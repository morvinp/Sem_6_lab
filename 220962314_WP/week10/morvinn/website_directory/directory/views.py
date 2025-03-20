# directory/views.py
from django.shortcuts import render, redirect
from .models import Category, Page
from .forms import CategoryForm, PageForm

# View to display all categories and pages
def directory_view(request):
    categories = Category.objects.all()
    return render(request, 'directory/directory.html', {'categories': categories})

# View to add a new category
def add_category(request):
    if request.method == 'POST':
        form = CategoryForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('directory_view')
    else:
        form = CategoryForm()
    return render(request, 'directory/add_category.html', {'form': form})

# View to add a new page under a category
def add_page(request):
    if request.method == 'POST':
        form = PageForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('directory_view')
    else:
        form = PageForm()
    return render(request, 'directory/add_page.html', {'form': form})
