from django.shortcuts import render, redirect
from .models import Works, Lives
from .forms import WorksForm, CompanySearchForm


def home(request):
    return render(request, 'home.html')  # Ensure you have 'home.html' in templates

def add_work_entry(request):
    if request.method == 'POST':
        form = WorksForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('home')  # Ensure 'home' exists in urls.py
    else:
        form = WorksForm()
    return render(request, 'add_work.html', {'form': form})
    
def search_employees(request):
    employees = None
    if request.method == "POST":
        form = CompanySearchForm(request.POST)
        if form.is_valid():
            company_name = form.cleaned_data['company_name']
            employees = Works.objects.filter(company_name=company_name).select_related()
    else:
        form = CompanySearchForm()
    return render(request, 'search_employees.html', {'form': form, 'employees': employees})
