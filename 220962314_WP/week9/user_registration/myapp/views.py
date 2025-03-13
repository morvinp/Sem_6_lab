from django.shortcuts import render, redirect
from .forms import RegisterForm

def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            contact_number = form.cleaned_data['contact_number']
            # Redirect to the success page with query parameters
            return redirect(f'/success/?username={username}&email={email}&contact_number={contact_number}')
    else:
        form = RegisterForm()

    return render(request, 'register.html', {'form': form})

def success(request):
    username = request.GET.get('username')
    email = request.GET.get('email')
    contact_number = request.GET.get('contact_number')
    
    return render(request, 'success.html', {'username': username, 'email': email, 'contact_number': contact_number})
