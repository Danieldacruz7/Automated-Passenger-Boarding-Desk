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

load_dotenv()

CONFIG = {
    'SUBSCRIPTION_KEY': os.getenv("SUBSCRIPTION_KEY"),
    'LOCATION': 'trial',
    'ACCOUNT_ID': os.getenv("ACCOUNT_ID"), 
    'FACIAL_RECOGNITION_ENDPOINT': os.getenv('AZURE_FACIAL_RECOGNIZER_ENDPOINT'), 
    'FACIAL_RECOGNITION_KEY': os.getenv('AZURE_FACIAL_RECOGNIZER_KEY'), 
    'FORM_RECOGNITION_ENDPOINT': os.getenv('AZURE_FORM_RECOGNIZER_ENDPOINT'), 
    'FORM_RECOGNITION_KEY': os.getenv('AZURE_FORM_RECOGNIZER_KEY')
}