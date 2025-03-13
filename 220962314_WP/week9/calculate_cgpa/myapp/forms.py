from django import forms

class CGPAForm(forms.Form):
    name = forms.CharField(max_length=100, required=True)
    total_marks = forms.IntegerField(min_value=0, required=True)
