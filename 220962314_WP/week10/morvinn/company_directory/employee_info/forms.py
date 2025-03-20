# employee_info/forms.py
from django import forms
from .models import Works, Lives

# Form to insert data into the WORKS table
class WorksForm(forms.ModelForm):
    class Meta:
        model = Works
        fields = ['person_name', 'company_name', 'salary']

# Form to search for people working at a particular company
class CompanySearchForm(forms.Form):
    company_name = forms.CharField(max_length=100)
