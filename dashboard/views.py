from django.shortcuts import render

def dashboard_index(request):
    return render(request, 'dashboard/index.html')
