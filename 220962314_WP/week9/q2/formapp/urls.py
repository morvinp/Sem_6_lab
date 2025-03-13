from django.urls import path
from .views import vote_view, submit_vote

urlpatterns = [
    path("", vote_view, name="vote"),
    path("submit-vote/", submit_vote, name="submit_vote"),
]
