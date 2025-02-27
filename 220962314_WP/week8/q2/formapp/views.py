from django.shortcuts import render, redirect

def first_page(request):
    subjects = ['Math', 'Science', 'History', 'English', 'Computer Science']

    if request.method == "POST":
        # Store form data in session
        request.session['name'] = request.POST.get('name', '')
        request.session['roll'] = request.POST.get('roll', '')
        request.session['subject'] = request.POST.get('subject', '')

        return redirect('second_page')  # Redirect to second page

    return render(request, 'firstPage.html', {'subjects': subjects})

def second_page(request):
    # Retrieve session data
    name = request.session.get('name', 'Unknown')
    roll = request.session.get('roll', 'Unknown')
    subject = request.session.get('subject', 'Unknown')

    return render(request, 'secondPage.html', {'name': name, 'roll': roll, 'subject': subject})
