# from django.shortcuts import render

# def index(request):
#     return render(request, 'basic.html')

from django.shortcuts import render

def index(request):
    student_details = ""
    percentage = 0.0

    if request.method == 'POST':
        name = request.POST.get('name')
        dob = request.POST.get('dob')
        address = request.POST.get('address')
        contact = request.POST.get('contact')
        email = request.POST.get('email')
        
        english_marks = int(request.POST.get('english_marks'))
        physics_marks = int(request.POST.get('physics_marks'))
        chemistry_marks = int(request.POST.get('chemistry_marks'))

        # Calculate the total marks and percentage
        total_marks = english_marks + physics_marks + chemistry_marks
        max_marks = 300  # Assuming each subject has a maximum of 100 marks
        percentage = (total_marks / max_marks) * 100

        # Create a student details string
        student_details = f"Name: {name}\nDate of Birth: {dob}\nAddress: {address}\nContact: {contact}\nEmail: {email}\nMarks (English): {english_marks}\nMarks (Physics): {physics_marks}\nMarks (Chemistry): {chemistry_marks}"

    return render(request, 'basic.html', {
        'student_details': student_details,
        'percentage': percentage,
    })
