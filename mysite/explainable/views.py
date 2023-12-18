from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic.edit import CreateView
from django.urls import reverse_lazy
from .models import UploadCSV, UploadMOD, UploadTrain


# Create your views here.
def process_data(ext, upload):
    ans = []
    for i in upload.objects.all(): 
        names = [i.upload_file.name for i in ans]
        if ext in i.upload_file.name and i.upload_file.name not in names:
            ans.append(i)
    return ans

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def performance(request):
    return HttpResponse("Performance")
def bias(request):
    return HttpResponse("Bias")
def enviromental(request):
    return HttpResponse("Enviromental")
def ethical(request):
    return HttpResponse("Ethical")



class UploadViewCSV(CreateView):
    model = UploadCSV
    fields = ['upload_file', ]
    success_url = reverse_lazy('fileupload')
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['documents'] = process_data("test.csv", UploadCSV)

        return context

class UploadViewMOD(CreateView):
    model = UploadMOD
    fields = ['upload_file', ]
    success_url = reverse_lazy('fileupload')
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["documents"] = process_data(".pkl", UploadMOD)
        print(context)
        return context

class UploadViewTrain(CreateView):
    model = UploadTrain
    fields = ['upload_file', ]
    success_url = reverse_lazy('fileupload')
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["documents"] = process_data("train.csv", UploadTrain)
        print(context)

        return context