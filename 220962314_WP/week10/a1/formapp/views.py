from django.shortcuts import render, get_object_or_404
from .models import Institute

def institute_list(request):
    institutes = Institute.objects.all()
    return render(request, 'institute_list.html', {'institutes': institutes})

def institute_details(request, institute_id):
    institute = get_object_or_404(Institute, pk=institute_id)
    return render(request, 'institute_details.html', {'institute': institute})
