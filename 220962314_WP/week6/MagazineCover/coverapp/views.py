from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage

def index(request):
    return render(request, 'coverapp/index.html')

def generate_cover(request):
    if request.method == 'POST':
        image = request.FILES['image']
        background_color = request.POST['background_color']
        font_size = request.POST['font_size']
        font_color = request.POST['font_color']
        title = request.POST['title']
        subtitle = request.POST['subtitle']

        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)  # This should now work correctly

        return render(request, 'coverapp/cover.html', {
            'uploaded_file_url': uploaded_file_url,
            'background_color': background_color,
            'font_size': font_size,
            'font_color': font_color,
            'title': title,
            'subtitle': subtitle,
        })
    return HttpResponse("Invalid request")