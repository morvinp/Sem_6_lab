from django.db import models

class Vote(models.Model):
    choice = models.CharField(max_length=20)
    count = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.choice}: {self.count}"
