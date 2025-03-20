from django import forms
from .models import Works

class WorksForm(forms.ModelForm):
    class Meta:
        model = Works
        fields = ['person_name', 'company_name', 'salary']

class CompanySearchForm(forms.Form):
    company_name = forms.CharField(label="Enter Company Name", max_length=100)
