import os
import io
import datetime
import requests
import pandas as pd
from io import BytesIO
from PIL import Image, ImageDraw
from urllib.parse import urlparse
import glob, os, sys, time, uuid
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import FormRecognizerClient
from video_indexer import VideoIndexer
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import TrainingStatusType
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.formrecognizer import FormTrainingClient
load_dotenv()

CONFIG = {
    'SUBSCRIPTION_KEY': os.getenv("SUBSCRIPTION_KEY"),
    'LOCATION': 'trial',
    'ACCOUNT_ID': os.getenv("ACCOUNT_ID"), 

    'FACIAL_RECOGNITION_ENDPOINT': os.getenv('AZURE_FACIAL_RECOGNIZER_ENDPOINT'), 
    'FACIAL_RECOGNITION_KEY': os.getenv('AZURE_FACIAL_RECOGNIZER_KEY'), 

    'FORM_RECOGNITION_ENDPOINT': os.getenv('AZURE_FORM_RECOGNIZER_ENDPOINT'), 
    'FORM_RECOGNITION_KEY': os.getenv('AZURE_FORM_RECOGNIZER_KEY'),

    'OBJECT_DETECTION_TRAINING_ENDPOINT' : os.getenv('OBJECT_DETECTION_TRAINING_ENDPOINT'), 
    'OBJECT_DETECTION_TRAINING_KEY' : os.getenv('OBJECT_DETECTION_TRAINING_KEY'),
    'OBJECT_DETECTION_TRAINING_RESOURCE_ID' : os.getenv('OBJECT_DETECTION_TRAINING_RESOURCE_ID'),

    'OBJECT_DETECTION_PREDICTION_ENDPOINT' : os.getenv('OBJECT_DETECTION_PREDICTION_ENDPOINT'),
    'OBJECT_DETECTION_PREDICTION_KEY' : os.getenv('OBJECT_DETECTION_PREDICTION_KEY'),
    'OBJECT_DETECTION_PREDICTION_RESOURCE_ID' : os.getenv('OBJECT_DETECTION_PREDICTION_RESOURCE_ID')
}

form_recognizer_client = FormRecognizerClient(endpoint=CONFIG['FORM_RECOGNIZER_ENDPOINT'], credential=AzureKeyCredential(CONFIG['FORM_RECOGNIZER_KEY']))
form_training_client = FormTrainingClient(endpoint=CONFIG['FORM_RECOGNITION_ENDPOINT'], credential=AzureKeyCredential(CONFIG['FORM_RECOGNIZER_KEY']))
face_client = FaceClient(CONFIG['FACIAL_RECOGNITION_ENDPOINT'], CognitiveServicesCredentials(CONFIG['FACIAL_RECOGNITION_ENDPOINT']))
training_credentials = ApiKeyCredentials(in_headers={"Training-key": CONFIG['OBJECT_DETECTION_TRAINING_KEY']})
trainer = CustomVisionTrainingClient(CONFIG['OBJECT_DETECTION_TRAINING_ENDPOINT'], training_credentials)

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": CONFIG['OBJECT_DETECTION_PREDICTION_KEY']})
predictor = CustomVisionPredictionClient(CONFIG['OBJECT_DETECTION_PREDICTION_ENDPOINT'], prediction_credentials)

video_analysis = VideoIndexer(
    vi_location=CONFIG['LOCATION'],
    vi_account_id=CONFIG['ACCOUNT_ID'], 
    vi_subscription_key=CONFIG['SUBSCRIPTION_KEY']
)