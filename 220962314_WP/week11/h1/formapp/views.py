from django.shortcuts import render, redirect
from .forms import AuthorForm, PublisherForm, BookForm
from .models import Book

def home(request):
    author_form = AuthorForm()
    publisher_form = PublisherForm()
    book_form = BookForm()
    
    if request.method == "POST":
        if 'author_submit' in request.POST:
            author_form = AuthorForm(request.POST)
            if author_form.is_valid():
                author_form.save()
                return redirect('home')

        elif 'publisher_submit' in request.POST:
            publisher_form = PublisherForm(request.POST)
            if publisher_form.is_valid():
                publisher_form.save()
                return redirect('home')

        elif 'book_submit' in request.POST:
            book_form = BookForm(request.POST)
            if book_form.is_valid():
                book_form.save()
                return redirect('home')

    books = Book.objects.all()  # Retrieve all books to display
    
    return render(request, 'home.html', {
        'author_form': author_form,
        'publisher_form': publisher_form,
        'book_form': book_form,
        'books': books
    })
