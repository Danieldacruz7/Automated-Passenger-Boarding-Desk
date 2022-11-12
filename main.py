import os
import io
import time
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
from utils import build_person_group, detect_faces, detect_face_from_any_url, list_all_faces_from_detected_face_object, perform_prediction

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

form_recognizer_client = FormRecognizerClient(endpoint=CONFIG['FORM_RECOGNITION_ENDPOINT'], credential=AzureKeyCredential(CONFIG['FORM_RECOGNITION_KEY']))
form_training_client = FormTrainingClient(endpoint=CONFIG['FORM_RECOGNITION_ENDPOINT'], credential=AzureKeyCredential(CONFIG['FORM_RECOGNITION_KEY']))
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

publish_iteration_name = "lighter-detection-model-v10"
project_id = "6bbf9a05-17c3-4f1f-acaf-05b4d1bafd69"
iteration_id = "7e20214d-3830-4a7a-8dee-9bcf1670f580"

boarding_dictionary = {
    "daniel da cruz": {
                            "digital id url": "https://udacityestorageaccount7.blob.core.windows.net/myblobcontainer7/ca-dl-daniel-da-cruz.png?sp=r&st=2022-11-11T04:20:07Z&se=2022-11-18T13:20:07Z&spr=https&sv=2021-06-08&sr=b&sig=%2BYr%2FowOPJfmdtr9Riz2j3QFsthTDrWKgjXqOgI4mBf0%3D", 
                            "Lighter image": "lighter_test_set_1of5.jpg",
                            "Carrier": "ZA"
    }

}

flight_manifest_dictonary = {
        "daniel da cruz": {
                            "Passenger Name": "Daniel da Cruz", 
                            "Date of Birth": datetime.date(1995, 8, 29),
                            "Carrier": "ZA", 
                            "Flight No.": 619, 
                            "Class": "A", 
                            "From": "Rustenburg", 
                            "To": "Cape Town", 
                            "Date": "November 11, 2022", 
                            "Baggage": "YES", 
                            "Seat": "30A", 
                            "Gate": "H2", 
                            "Boarding Time": "11:00 PM CAT", 
                            "Ticket No.": "ETK-737268572620C", 
                            "DoB Validation": False, 
                            "PersonValidation": False, 
                            "LuggageValidation": False, 
                            "NameValidation": False, 
                            "BoardingPassValidation": False
    }, 

        'helena da cruz': {
                            "Passenger Name": "Helena da Cruz", 
                            "Date of Birth": datetime.date(2000, 4, 7),
                            "Carrier": "ZA", 
                            "Flight No.": 619, 
                            "Class": "A", 
                            "From": "Rustenburg", 
                            "To": "Cape Town", 
                            "Date": "November 11, 2022", 
                            "Baggage": "YES", 
                            "Seat": "31A", 
                            "Gate": "H2", 
                            "Boarding Time": "11:00 PM CAT", 
                            "Ticket No.": "ETK-737268572620C", 
                            "DoB Validation": False, 
                            "PersonValidation": False, 
                            "LuggageValidation": False, 
                            "NameValidation": False, 
                            "BoardingPassValidation": False
                }, 
        'john doe': {
                            "Passenger Name": "John Doe", 
                            "Date of Birth": datetime.date(1980, 2, 5),
                            "Carrier": "ZA", 
                            "Flight No.": 619, 
                            "Class": "A", 
                            "From": "Johannesburg", 
                            "To": "Cape Town", 
                            "Date": "November 11, 2022", 
                            "Baggage": "YES", 
                            "Seat": "40A", 
                            "Gate": "H2", 
                            "Boarding Time": "11:00 PM CAT", 
                            "Ticket No.": "ETK-737268572620C", 
                            "DoB Validation": False, 
                            "PersonValidation": False, 
                            "LuggageValidation": False, 
                            "NameValidation": False, 
                            "BoardingPassValidation": False
    }, 
        'mark musk': {
                            "Passenger Name": "Mark Musk", 
                            "Date of Birth": datetime.date(1989, 2, 8),
                            "Carrier": "ZA", 
                            "Flight No.": 420, 
                            "Class": "E", 
                            "From": "New York", 
                            "To": "Austin", 
                            "Date": "November 20, 2022", 
                            "Baggage": "YES", 
                            "Seat": "15F", 
                            "Gate": "I2", 
                            "Boarding Time": "12:00 PM PST", 
                            "Ticket No.": "ETK-737268572620C", 
                            "DoB Validation": False, 
                            "PersonValidation": False, 
                            "LuggageValidation": False, 
                            "NameValidation": False, 
                            "BoardingPassValidation": False
    }, 
        'noah taleb': {
                            "Passenger Name": "Noah Taleb", 
                            "Date of Birth": datetime.date(1968, 2, 8),
                            "Carrier": "ZA", 
                            "Flight No.": 820, 
                            "Class": "D", 
                            "From": "New York", 
                            "To": "San Francisco", 
                            "Date": "November 15, 2022", 
                            "Baggage": "YES", 
                            "Seat": "24B", 
                            "Gate": "I2", 
                            "Boarding Time": "12:00 PM PST", 
                            "Ticket No.": "ETK-737268572620C", 
                            "DoB Validation": False, 
                            "PersonValidation": False, 
                            "LuggageValidation": False, 
                            "NameValidation": False, 
                            "BoardingPassValidation": False
    }

}
digital_id_directory = "./data/digital_id_template/Test-Images/ca-dl-"
custom_boarding_pass_id = "3101438c-b68f-4695-8a70-97e3eef7121a"
boarding_pass_directory = "./data/boarding_pass_template/Test-Images/"
digital_video_directory = "./data/digital-video-sample/"
thumbnail_directory = "./data/ai-generated-thumbnails/"

print("Hello welcome to the Airport of the future! ")
first_name = input("Please enter your first name: ")
second_name = input("Please enter your second name: ")
full_name = first_name + " " + second_name
altered_full_name = "-".join(full_name.lower().split())

if full_name in flight_manifest_dictonary:
    print("Please present your ID")
    time.sleep(5)
    # Extract digital ID information
    with open(digital_id_directory + altered_full_name + ".png", "rb") as test_data:
                digital_id_info = form_recognizer_client.begin_recognize_identity_documents(test_data, content_type="image/png")
    digital_id_results = digital_id_info.result()

    # Extract boarding pass information
    with open(boarding_pass_directory + altered_full_name + ".pdf", "rb") as test_data:
                boarding_pass_info = form_recognizer_client.begin_recognize_custom_forms(model_id=custom_boarding_pass_id, form = test_data, content_type='application/pdf')
    boarding_pass_results = boarding_pass_info.result()

    # Extract facial features from video 
    uploaded_video_id = video_analysis.upload_to_video_indexer(
      input_filename=digital_video_directory + altered_full_name + ".mp4",
      video_name=altered_full_name + "-boarding-pass",  # unique identifier for video in Video Indexer platform
      video_language='English'
    )
    print("Please wait as we analyze your video...")
    time.sleep(30)
    print("Analysis complete.")
    video_info = video_analysis.get_video_info(uploaded_video_id, video_language='English')

    images = []
    img_raw = []
    img_strs = []
    thumbnails = []
    for each_thumb in video_info['videos'][0]['insights']['faces'][0]['thumbnails']:
        if 'fileName' in each_thumb and 'id' in each_thumb:
            file_name = each_thumb['fileName']
            thumb_id = each_thumb['id']
            img_code = video_analysis.get_thumbnail_from_video_indexer(uploaded_video_id,  thumb_id)
            img_strs.append(img_code)
            img_stream = io.BytesIO(img_code)
            img_raw.append(img_stream)
            img = Image.open(img_stream)
            images.append(img)
            thumbnails.append(thumb_id)

    name = video_info['name']
    j = 1
    for img in images:
        img.save(thumbnail_directory + "{}".format(name) + '/human-face' + str(j) + '.jpg')
        j +=1

    # Build person group id
    person_group_id = str(uuid.uuid4())
    build_person_group(face_client, person_group_id, altered_full_name, "./data/ai-generated-thumbnails/{}/".format(altered_full_name + "-boarding-pass"))
    test_images = [[file for file in glob.glob('./data/ai-generated-thumbnails/{}/*.jpg'.format(altered_full_name + "-boarding-pass"))][-1]]
    person_group_face_id = detect_faces(face_client, test_images)

    # Detect face 
    source_faces_object = detect_face_from_any_url(face_client, digital_id_directory + altered_full_name + ".png")
    detected_face = list_all_faces_from_detected_face_object(source_faces_object)
    detected_face_id = detected_face[0].face_id

    # Prediction on lighter image
    local_image_path = r'data/lighter_test_images'
    with open(os.path.join (local_image_path,  boarding_dictionary["daniel da cruz"]['Lighter image']), "rb") as test_data:
        results = predictor.detect_image(project_id, publish_iteration_name, test_data.read())
        lighter_probs = results.predictions[0].probability
    
    # Flight manifest
    flight_manifest = pd.DataFrame(columns=["Passenger Name", "Date of Birth", "Carrier", "Flight No.", "Class", "From", "To", "Date", "Baggage", "Seat", "Gate", "Boarding Time", "Ticket No.", "DoB Validation", "PersonValidation", "LuggageValidation", "NameValidation", "BoardingPassValidation"]) 
    flight_manifest_list = [flight_manifest_dictonary["daniel da cruz"], flight_manifest_dictonary, flight_manifest_dictonary, flight_manifest_dictonary, flight_manifest_dictonary]
    for i in flight_manifest_list: 
        flight_manifest = flight_manifest.append(i, ignore_index=True)
    
    # Flight Manifest Validation
    for i in range(len(flight_manifest)):
    # Name Validation
        if (flight_manifest.loc[i, 'Passenger Name'].lower() == (digital_id_results['FirstName'] + " " + digital_id_results['LastName']).lower()) \
        and (flight_manifest.loc[i, 'Passenger Name'].lower() == (boarding_pass_results['Passenger Name'].lower())):
            flight_manifest.loc[i, 'NameValidation'] = True  
        
        # Date of Birth Validation: 
        if (flight_manifest.loc[i, 'Date of Birth'] == (digital_id_results['DateOfBirth'])):
            flight_manifest.loc[i, 'DoB Validation'] = True 
        
        # Boarding Pass Validation
        if (flight_manifest.loc[i, 'Carrier'] == boarding_pass_results['Flight Carrier']) \
        and (flight_manifest.loc[i, 'Flight No.'] == int(boarding_pass_results['Flight Number'])) \
        and (flight_manifest.loc[i, 'Class'] == boarding_pass_results['Flight Class']) \
        and (flight_manifest.loc[i, 'From'] == boarding_pass_results['Departure Location']) \
        and (flight_manifest.loc[i, 'To'] == boarding_pass_results['Arrival Location']) \
        and (flight_manifest.loc[i, 'Date'] == boarding_pass_results['Date']) \
        and (flight_manifest.loc[i, 'Baggage'] == boarding_pass_results['Baggage Allowance']) \
        and (flight_manifest.loc[i, 'Seat'] == boarding_pass_results['Seat Allocation']) \
        and (flight_manifest.loc[i, 'Gate'] == boarding_pass_results['Boarding Gate']) \
        and (flight_manifest.loc[i, 'Boarding Time'] == boarding_pass_results['Boarding Time.']) \
        and (flight_manifest.loc[i, 'Ticket No.'] == boarding_pass_results['Ticket Number']):
            flight_manifest.loc[i, 'BoardingPassValidation'] = True

        # Person Validation
        verify_result_same = face_client.face.verify_face_to_face(detected_face_id, person_group_face_id)
        if verify_result_same.is_identical:
                flight_manifest.loc[i, 'PersonValidation'] = True 

        # Luggage Validation
        if lighter_probs  < 0.6:
            flight_manifest.loc[i, 'LuggageValidation'] = True

else:
    print("Sorry this name does not appear on the flight bookings list!")
