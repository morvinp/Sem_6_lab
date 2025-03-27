from django.shortcuts import render, redirect
from .models import Student
from .forms import StudentForm

def student_view(request):
    if request.method == "POST":
        form = StudentForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('student_view')  # Refresh page to show updated list

    else:
        form = StudentForm()

    students = Student.objects.all()  # Retrieve all student records

    return render(request, 'student_form.html', {'form': form, 'students': students})
