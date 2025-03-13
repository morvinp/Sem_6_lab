from django.shortcuts import render
from django.http import JsonResponse

# Dictionary to store votes (reset on server restart)
vote_counts = {
    "Good": 0,
    "Satisfactory": 0,
    "Bad": 0
}

# Render the voting page
def vote_view(request):
    return render(request, "vote.html", {"votes": vote_counts})

# Handle vote submission
def submit_vote(request):
    if request.method == "POST":
        choice = request.POST.get("choice")
        if choice in vote_counts:
            vote_counts[choice] += 1  # Update the vote count

        # Calculate percentages
        total_votes = sum(vote_counts.values())
        results = {k: round((v / total_votes) * 100, 2) if total_votes > 0 else 0 for k, v in vote_counts.items()}

        return JsonResponse({"results": results})

    return JsonResponse({"error": "Invalid request"}, status=400)
