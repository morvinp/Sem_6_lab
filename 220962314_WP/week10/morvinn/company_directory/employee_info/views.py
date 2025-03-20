# employee_info/views.py
from django.shortcuts import render, redirect
from .models import Works, Lives
from .forms import WorksForm, CompanySearchForm

# View to add a person to the WORKS table
def add_work(request):
    if request.method == 'POST':
        form = WorksForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('add_work')
    else:
        form = WorksForm()
    return render(request, 'employee_info/add_work.html', {'form': form})

# View to search for people working at a particular company and retrieve cities
def search_company(request):
    if request.method == 'POST':
        form = CompanySearchForm(request.POST)
        if form.is_valid():
            company_name = form.cleaned_data['company_name']
            # Retrieve people working at the company
            works = Works.objects.filter(company_name=company_name)
            people_cities = []
            for work in works:
                person_name = work.person_name
                # Find the city of each person
                try:
                    city = Lives.objects.get(person_name=person_name).city
                    people_cities.append((person_name, city))
                except Lives.DoesNotExist:
                    people_cities.append((person_name, "Unknown"))
            return render(request, 'employee_info/search_results.html', {'people_cities': people_cities, 'company_name': company_name})
    else:
        form = CompanySearchForm()
    return render(request, 'employee_info/search_company.html', {'form': form})
