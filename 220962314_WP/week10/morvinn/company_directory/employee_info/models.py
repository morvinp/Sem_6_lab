# employee_info/models.py
from django.db import models

class Works(models.Model):
    person_name = models.CharField(max_length=100)
    company_name = models.CharField(max_length=100)
    salary = models.FloatField()

    def __str__(self):
        return f"{self.person_name} works at {self.company_name}"

class Lives(models.Model):
    person_name = models.CharField(max_length=100)
    street = models.CharField(max_length=100)
    city = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.person_name} lives on {self.street}, {self.city}"
