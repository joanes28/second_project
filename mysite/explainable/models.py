from django.db import models
from django.conf import settings
import os
from django.core.exceptions import ValidationError
def validate_file_extension_csv(value, extensions):
    #extensions = [".csv"]
    ext = os.path.splitext(value.name)[1]
    if not (ext in extensions):
        raise ValidationError(u'File not supported!!')
"""
def validate_file_extension_mod(value):
    extensions = [".pkl"]
    ext = os.path.splitext(value.name)[1]
    if not (ext in extensions):
        raise ValidationError(u'File not supported!!')
"""

def valCSV(value):
    return validate_file_extension_csv(value, [".csv"])
def valPKL(value):
    return validate_file_extension_csv(value, [".pkl"])


def upload(name):
    if os.path.exists(os.path.join(settings.MEDIA_ROOT, name)):
         os.remove(os.path.join(settings.MEDIA_ROOT, name))   
    return name

def uploadtest(a, b):
    return upload('test.csv')
def uploadmodel(a, b):
    return upload('model.pkl')
def uploadtrain(a, b):
    return upload('train.csv')
# Create your models here.
class UploadCSV(models.Model):
    upload_file = models.FileField("Only upload CSV files", upload_to=uploadtest, validators = [valCSV])    
    upload_date = models.DateTimeField(auto_now_add =True)

class UploadMOD(models.Model):
    upload_file = models.FileField("Only upload PKL models", upload_to=uploadmodel, validators = [valPKL])    
    upload_date = models.DateTimeField(auto_now_add =True)

class UploadTrain(models.Model):
    upload_file = models.FileField("Only upload CSV files", upload_to=uploadtrain, validators = [valCSV])    
    upload_date = models.DateTimeField(auto_now_add =True)