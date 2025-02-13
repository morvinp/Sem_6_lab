from django.shortcuts import render
from datetime import datetime

# View for displaying the employee promotion form
def promotion_form(request):
    eligibility_result = None  # Default result, no eligibility check yet

    if request.method == "POST":
        # Get data from the form submission
        employee_id = request.POST.get('employeeId')
        date_of_joining = request.POST.get('doj')

        if date_of_joining:
            # Calculate the years of experience
            doj = datetime.strptime(date_of_joining, "%Y-%m-%d")
            current_date = datetime.now()
            years_of_experience = current_date.year - doj.year

            # Check if the employee has more than 5 years of experience
            if years_of_experience >= 5:
                eligibility_result = "YES"
                eligibility_class = "green"  # For green color
            else:
                eligibility_result = "NO"
                eligibility_class = "red"  # For red color
        else:
            eligibility_result = "Please provide your date of joining."
            eligibility_class = "orange"  # For orange color indicating an error

        # Pass the result to the template
        return render(request, 'employee_form.html', {
            'eligibility_result': eligibility_result,
            'eligibility_class': eligibility_class,
            'employee_id': employee_id,
            'doj': date_of_joining,
        })

    # If it's a GET request, just render the form without any eligibility result
    return render(request, 'employee_form.html')
