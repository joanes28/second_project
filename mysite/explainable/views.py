from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic.edit import CreateView
from django.urls import reverse_lazy
from .models import UploadCSV, UploadMOD, UploadTrain
from .pipeline.metrics import performance_evaluation 
from .pipeline.enviromental import inference_time_and_environmental_impact_evaluation 
from .pipeline.bias import model_bias_and_variance_analysis
from .pipeline.ethical import ethical_input_feature_analysis
from sklearn.utils.validation import check_is_fitted
import pickle
from django.conf import settings
import pandas as pd
import os
from  sklearn.exceptions import NotFittedError
from sklearn.base import is_classifier, is_regressor
import json
import shap

# Create your views here.
def openModel():
    with open(os.path.join(settings.MEDIA_ROOT, "model.pkl"), "rb")as f:
        model = pickle.load(f)
    try:
        check_is_fitted(model)
    except NotFittedError as e:
        try:
            train = pd.read_csv(os.path.join(settings.MEDIA_ROOT, "train.csv"))
            y = train["target"]
            X = train.drop(["target"], axis =1)
            model.fit(X, y)
            pickle.dump(model, open(os.path.join(settings.MEDIA_ROOT, "model.pkl"), "wb"))
        except:
            return False
    return True

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
    test = pd.read_csv(os.path.join(settings.MEDIA_ROOT, "test.csv"))
    if openModel():
        with open(os.path.join(settings.MEDIA_ROOT, "model.pkl"), "rb")as f:
            model = pickle.load(f)
    else:
        return HttpResponse("MODEL IS NOT FITTED AND NO CSV FOR TRAINING HAS BEEN PROVIDED")

    y = test["target"]
    X = test.drop(["target"], axis = 1)
    performance = performance_evaluation(model, X, y )

    if is_classifier(model):
        performance = performance.classification_metrics()
    else:
        performance = performance.regression_metrics()
    return render(request, "explainable/index.html", context = {"content": performance})


def bias(request):

    test = pd.read_csv(os.path.join(settings.MEDIA_ROOT, "test.csv"))
    if openModel():
        with open(os.path.join(settings.MEDIA_ROOT, "model.pkl"), "rb")as f:
            model = pickle.load(f)
    else:
        return HttpResponse("MODEL IS NOT FITTED AND NO CSV FOR TRAINING HAS BEEN PROVIDED")

    y = test["target"]
    X = test.drop(["target"], axis = 1)

    bias = model_bias_and_variance_analysis(model, X, y )
    if is_classifier(model):
        bias = bias.model_bias_and_variance_analysis_classification()
    else:
        bias = bias.model_bias_and_variance_analysis_regression()

    return render(request, "explainable/bias.html", context = {"content": bias})



def enviromental(request):
    test = pd.read_csv(os.path.join(settings.MEDIA_ROOT, "test.csv"))
    y = test["target"]
    X = test.drop(["target"], axis = 1)

    if openModel():
        with open(os.path.join(settings.MEDIA_ROOT, "model.pkl"), "rb")as f:
            model = pickle.load(f)
    else:
        return HttpResponse("MODEL IS NOT FITTED AND NO CSV FOR TRAINING HAS BEEN PROVIDED")

    results = inference_time_and_environmental_impact_evaluation(model, X)
    return render(request, "explainable/index.html", context = {"content": results})
def ethical(request):

    test = pd.read_csv(os.path.join(settings.MEDIA_ROOT, "test.csv"))
    y = test["target"]
    X = test.drop(["target"], axis = 1)

    if openModel():
        with open(os.path.join(settings.MEDIA_ROOT, "model.pkl"), "rb")as f:
            model = pickle.load(f)
    else:
        return HttpResponse("MODEL IS NOT FITTED AND NO CSV FOR TRAINING HAS BEEN PROVIDED")

    analysis = ethical_input_feature_analysis(model, X)
    return render(request, "explainable/ethical.html", context = {"content": analysis})


class UploadViewCSV(CreateView):
    model = UploadCSV
    fields = ['upload_file', ]
    success_url = reverse_lazy('testupload')
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['documents'] = process_data("test.csv", UploadCSV)

        return context

class UploadViewMOD(CreateView):
    model = UploadMOD
    fields = ['upload_file', ]
    success_url = reverse_lazy('modelupload')
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["documents"] = process_data(".pkl", UploadMOD)
        print(context)
        return context

class UploadViewTrain(CreateView):
    model = UploadTrain
    fields = ['upload_file', ]
    success_url = reverse_lazy('trainupload')
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["documents"] = process_data("train.csv", UploadTrain)
        print(context)

        return context