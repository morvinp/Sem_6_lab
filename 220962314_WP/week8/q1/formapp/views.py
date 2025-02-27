from django.shortcuts import render

def index(request):
    manufacturers = ['Toyota', 'Ford', 'Honda', 'BMW', 'Mercedes']
    return render(request, 'index.html', {'manufacturers': manufacturers})

def display(request):
    if request.method == 'GET':
        manufacturer = request.GET.get('manufacturer', 'Unknown')
        model = request.GET.get('model', 'Unknown')
        return render(request, 'display.html', {'manufacturer': manufacturer, 'model': model})
