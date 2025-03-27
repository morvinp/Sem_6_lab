from django.shortcuts import render, redirect, get_object_or_404
from .models import Human
from .forms import HumanForm
from django.http import JsonResponse

def home(request):
    humans = Human.objects.all()  # Load all first names for dropdown
    return render(request, 'home.html', {'humans': humans})

def get_human_details(request, human_id):
    """Fetch details of the selected human and return as JSON."""
    human = get_object_or_404(Human, id=human_id)
    data = {
        'first_name': human.first_name,
        'last_name': human.last_name,
        'phone': human.phone,
        'address': human.address,
        'city': human.city
    }
    return JsonResponse(data)

def update_human(request, human_id):
    """Update human details from the form submission."""
    human = get_object_or_404(Human, id=human_id)
    
    if request.method == "POST":
        form = HumanForm(request.POST, instance=human)
        if form.is_valid():
            form.save()
            return redirect('home')
    
    return JsonResponse({'status': 'error', 'message': 'Invalid data'})

def delete_human(request, human_id):
    """Delete the selected human and refresh the dropdown."""
    human = get_object_or_404(Human, id=human_id)
    human.delete()
    return JsonResponse({'status': 'success'})
