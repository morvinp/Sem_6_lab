from django.shortcuts import render, redirect
from .forms import CGPAForm

def calculate(request):
    if request.method == 'POST':
        form = CGPAForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            total_marks = form.cleaned_data['total_marks']
            
            # Store data in session
            request.session['name'] = name
            request.session['total_marks'] = total_marks
            
            # Redirect to the second page
            return redirect('result')
    else:
        form = CGPAForm()

    return render(request, 'calculate.html', {'form': form})

def result(request):
    # Retrieve data from session
    name = request.session.get('name')
    total_marks = request.session.get('total_marks')
    
    if name and total_marks is not None:
        cgpa = total_marks / 50  # CGPA calculation
        return render(request, 'result.html', {'name': name, 'cgpa': cgpa})
    else:
        return redirect('calculate')
