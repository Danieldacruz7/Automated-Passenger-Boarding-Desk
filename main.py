import os, io, time, datetime
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
from utils import build_person_group, detect_faces, detect_face_from_any_url, list_all_faces_from_detected_face_object
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
face_client = FaceClient(CONFIG['FACIAL_RECOGNITION_ENDPOINT'], CognitiveServicesCredentials(CONFIG['FACIAL_RECOGNITION_KEY']))
training_credentials = ApiKeyCredentials(in_headers={"Training-key": CONFIG['OBJECT_DETECTION_TRAINING_KEY']})
trainer = CustomVisionTrainingClient(CONFIG['OBJECT_DETECTION_TRAINING_ENDPOINT'], training_credentials)

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": CONFIG['OBJECT_DETECTION_PREDICTION_KEY']})
predictor = CustomVisionPredictionClient(CONFIG['OBJECT_DETECTION_PREDICTION_ENDPOINT'], prediction_credentials)

video_analysis = VideoIndexer(
    vi_location=CONFIG['LOCATION'],
    vi_account_id=CONFIG['ACCOUNT_ID'], 
    vi_subscription_key=CONFIG['SUBSCRIPTION_KEY']
)

lighter_detection_dictionary = {
    "publish_iteration_name" : "lighter-detection-model-v10",
    "project_id" : "6bbf9a05-17c3-4f1f-acaf-05b4d1bafd69",
    "iteration_id" : "7e20214d-3830-4a7a-8dee-9bcf1670f580" 
}

boarding_dictionary = {
    "daniel da cruz": {"Lighter image": "lighter_test_set_1of5.jpg"}, 
    "helena da cruz": {"Lighter image": "lighter_test_set_2of5.jpg"}, 
    "john doe": {"Lighter image": "lighter_test_set_3of5.jpg"}, 
    "mark musk": {"Lighter image": "lighter_test_set_4of5.jpg"}, 
    "noah taleb": {"Lighter image": "lighter_test_set_5of5.jpg"}
    }

response_dictionary = {
    "successful response": "Dear {}, \nYou are welcome to flight # {} leaving at {} from {} to {}. \nYour seat number is {}, and it is confirmed. \nWe did not find a prohibited item (lighter) in your carry-on baggage, \nthanks for following the procedure. \nYour identity is verified so please board the plane. ",
    "failed response": "Dear Sir/Madam, \nSome of the information in your boarding pass does not match the flight manifest data, so you cannot board the plane. \nPlease see a customer service representative.",
    "Date of birth incorrect response": "Dear Sir/Madam, \nThe date of birth on your ID card does not match the flight manifest data, so you cannot board the plane. \nPlease see a customer service representative.",
    "lighter found response": "Dear {}, \nYou are welcome to flight # {} leaving at {} from {} to {}. \nYour seat number is {}, and it is confirmed. \nWe have found a prohibited item in your carry-on baggage, and it is flagged for removal. \nYour identity is verified. However, your baggage verification failed, so please see a customer service representative.",
    "face identification failed response": "Dear {}, \nYou are welcome to flight # {} leaving at {} from {} to {}. \nYour seat number is {}, and it is confirmed. \nWe did not find a prohibited item (lighter) in your carry-on baggage. \nThanks for following the procedure. \nYour identity could not be verified. Please see a customer service representative.",
    "boarding pass validation failed response": "Dear Sir/Madam, \nSome of the information in your boarding pass does not match the flight manifest data, so you cannot board the plane. \nPlease see a customer service representative.", 
    "ID validation failed response": "Dear Sir/Madam, \nSome of the information on your ID card does not match the flight manifest data, so you cannot board the plane. \nPlease see a customer service representative."
}

flight_manifest_dictonary = {
        "daniel da cruz": {
                            "Passenger Name": "daniel da cruz", 
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
                            "Passenger Name": "helena da cruz", 
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
                            "Passenger Name": "john doe", 
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
                            "Passenger Name": "mark musk", 
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
                            "Passenger Name": "noah taleb", 
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

# Flight manifest
import os.path
if os.path.exists('flight_manifest.csv') == True: 
    flight_manifest = pd.read_csv('flight_manifest.csv')
    flight_manifest["Date of Birth"] = pd.to_datetime(flight_manifest["Date of Birth"], format="%Y/%m/%d")
else:
    flight_manifest = pd.DataFrame(columns=["Passenger Name", "Date of Birth", "Carrier", "Flight No.", "Class", "From", "To", "Date", "Baggage", "Seat", "Gate", "Boarding Time", "Ticket No.", "DoB Validation", "PersonValidation", "LuggageValidation", "NameValidation", "BoardingPassValidation"]) 
    flight_manifest_list = [flight_manifest_dictonary[key] for key in flight_manifest_dictonary.keys()]
    for i in flight_manifest_list: 
        flight_manifest = flight_manifest.append(i, ignore_index=True)
    flight_manifest["Date of Birth"] = pd.to_datetime(flight_manifest["Date of Birth"], format="%Y/%m/%d")

print("Hello welcome to the Airport of the future! ")
first_name = input("Please enter your first name: ")
second_name = input("Please enter your second name: ")
full_name = (first_name.strip() + " " + second_name.strip()).lower()
altered_full_name = "-".join(full_name.lower().split())

if full_name in flight_manifest_dictonary:
    print("")
    print("Please present your ID")
    # Extract digital ID information
    with open(digital_id_directory + altered_full_name + ".png", "rb") as test_data:
                digital_id_info = form_recognizer_client.begin_recognize_identity_documents(test_data, content_type="image/png")
    digital_id_results = digital_id_info.result()
    print("Thank you for presenting your ID. Proceed to show your boarding pass.")
    print("")
    
    # Extract boarding pass information
    print("Analyzing boarding pass...")
    with open(boarding_pass_directory + altered_full_name + ".pdf", "rb") as test_data:
                boarding_pass_info = form_recognizer_client.begin_recognize_custom_forms(model_id=custom_boarding_pass_id, form = test_data, content_type='application/pdf')
    boarding_pass_results = boarding_pass_info.result()
    print("Boarding pass successfully analyzed.")
    print("")

    # Extract facial features from video 
    print("Please look at the screen. The camera will now capture facial features...")
    uploaded_video_id = video_analysis.upload_to_video_indexer(
      input_filename=digital_video_directory + altered_full_name + ".mp4",
      video_name=altered_full_name + "-boarding-pass",  # unique identifier for video in Video Indexer platform
      video_language='English'
    )

    print("Please wait as we analyze your video...")
    video_info = video_analysis.get_video_info(uploaded_video_id, video_language='English')
    while video_info['state'] != 'Processed':
        #time.sleep(180) # Time to allow video processing to complete, and this is precautionary
        video_info = video_analysis.get_video_info(uploaded_video_id, video_language='English')
        print(video_info['state'])

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
    print("Analysis complete.")
    print("")

    # Prediction on lighter image
    print("Please proceed to baggage declaration...")
    local_image_path = r'data/lighter_test_images'
    with open(os.path.join (local_image_path,  boarding_dictionary[full_name]['Lighter image']), "rb") as test_data:
        results = predictor.detect_image(lighter_detection_dictionary["project_id"], lighter_detection_dictionary["publish_iteration_name"], test_data.read())
    lighter_probs = results.predictions[0].probability
    
    # Flight Manifest Validation
    flight_manifest_person_index = flight_manifest[flight_manifest['Passenger Name'] == full_name].index[0]
    # Name Validation
    if (flight_manifest.loc[flight_manifest_person_index, 'Passenger Name'].lower() == (digital_id_results[0].fields['FirstName'].value + " " + digital_id_results[0].fields['LastName'].value).lower()) \
    and (flight_manifest.loc[flight_manifest_person_index, 'Passenger Name'].lower() == ((boarding_pass_results[0].fields['Passenger Name'].value).lower())):
        flight_manifest.loc[flight_manifest_person_index, 'NameValidation'] = True  
        
    # Date of Birth Validation: 
    if (flight_manifest.loc[flight_manifest_person_index, 'Date of Birth'] == (digital_id_results[0].fields['DateOfBirth'].value)):
        flight_manifest.loc[flight_manifest_person_index, 'DoB Validation'] = True 
        
        # Boarding Pass Validation
    if (flight_manifest.loc[flight_manifest_person_index, 'Carrier'] == boarding_pass_results[0].fields['Flight Carrier'].value) \
    and (flight_manifest.loc[flight_manifest_person_index, 'Flight No.'] == int(boarding_pass_results[0].fields['Flight Number'].value)) \
    and (flight_manifest.loc[flight_manifest_person_index, 'Class'] == boarding_pass_results[0].fields['Flight Class'].value) \
    and (flight_manifest.loc[flight_manifest_person_index, 'From'] == boarding_pass_results[0].fields['Departure Location'].value) \
    and (flight_manifest.loc[flight_manifest_person_index, 'To'] == boarding_pass_results[0].fields['Arrival Location'].value) \
    and (flight_manifest.loc[flight_manifest_person_index, 'Date'] == boarding_pass_results[0].fields['Date'].value) \
    and (flight_manifest.loc[flight_manifest_person_index, 'Baggage'] == boarding_pass_results[0].fields['Baggage Allowance'].value) \
    and (flight_manifest.loc[flight_manifest_person_index, 'Seat'] == boarding_pass_results[0].fields['Seat Allocation'].value) \
    and (flight_manifest.loc[flight_manifest_person_index, 'Gate'] == boarding_pass_results[0].fields['Boarding Gate'].value) \
    and (flight_manifest.loc[flight_manifest_person_index, 'Boarding Time'] == boarding_pass_results[0].fields['Boarding Time.'].value) \
    and (flight_manifest.loc[flight_manifest_person_index, 'Ticket No.'] == boarding_pass_results[0].fields['Ticket Number'].value):
        flight_manifest.loc[flight_manifest_person_index, 'BoardingPassValidation'] = True

        # Person Validation
    verify_result_same = face_client.face.verify_face_to_face(detected_face_id, person_group_face_id)
    if verify_result_same.is_identical:
            flight_manifest.loc[flight_manifest_person_index, 'PersonValidation'] = True 

        # Luggage Validation
    if lighter_probs  < 0.6:
        flight_manifest.loc[flight_manifest_person_index, 'LuggageValidation'] = True
    
    # Validation Check
    count = 0
    for column in flight_manifest:
        if  flight_manifest.loc[flight_manifest_person_index, column] == False:
            count += 1
    
    if count >= 2:
        print(response_dictionary['failed response'])

    else:
        if flight_manifest.loc[flight_manifest_person_index, "DoB Validation":"BoardingPassValidation"].all() == True:
            print(response_dictionary["successful response"].format(first_name.capitalize(), 
                                                                    boarding_pass_results[0].fields['Flight Number'].value, 
                                                                    boarding_pass_results[0].fields['Boarding Time.'].value, 
                                                                    boarding_pass_results[0].fields['Departure Location'].value, 
                                                                    boarding_pass_results[0].fields['Arrival Location'].value, 
                                                                    boarding_pass_results[0].fields['Seat Allocation'].value))
        
        if flight_manifest.loc[flight_manifest_person_index, "DoB Validation"] == False:
            print(response_dictionary["Date of birth incorrect response"].format(first_name.capitalize(), 
                                                                            boarding_pass_results[0].fields['Flight Number'].value, 
                                                                            boarding_pass_results[0].fields['Boarding Time.'].value, 
                                                                            boarding_pass_results[0].fields['Departure Location'].value, 
                                                                            boarding_pass_results[0].fields['Arrival Location'].value, 
                                                                            boarding_pass_results[0].fields['Seat Allocation'].value))
            
        if flight_manifest.loc[flight_manifest_person_index, "NameValidation"] == False:
            print(response_dictionary["ID validation failed response"].format(first_name.capitalize(), 
                                                                            boarding_pass_results[0].fields['Flight Number'].value, 
                                                                            boarding_pass_results[0].fields['Boarding Time.'].value, 
                                                                            boarding_pass_results[0].fields['Departure Location'].value, 
                                                                            boarding_pass_results[0].fields['Arrival Location'].value, 
                                                                            boarding_pass_results[0].fields['Seat Allocation'].value))
        
        if flight_manifest.loc[flight_manifest_person_index, "PersonValidation"] == False:
            print(response_dictionary["face identification failed response"].format(first_name.capitalize(), 
                                                                    boarding_pass_results[0].fields['Flight Number'].value, 
                                                                    boarding_pass_results[0].fields['Boarding Time.'].value, 
                                                                    boarding_pass_results[0].fields['Departure Location'].value, 
                                                                    boarding_pass_results[0].fields['Arrival Location'].value, 
                                                                    boarding_pass_results[0].fields['Seat Allocation'].value))
        
        if flight_manifest.loc[flight_manifest_person_index, "BoardingPassValidation"] == False:
            print(response_dictionary["boarding pass validation failed response"].format(first_name.capitalize(), 
                                                                    boarding_pass_results[0].fields['Flight Number'].value, 
                                                                    boarding_pass_results[0].fields['Boarding Time.'].value, 
                                                                    boarding_pass_results[0].fields['Departure Location'].value, 
                                                                    boarding_pass_results[0].fields['Arrival Location'].value, 
                                                                    boarding_pass_results[0].fields['Seat Allocation'].value))
        
        if flight_manifest.loc[flight_manifest_person_index, "LuggageValidation"] == False:
            print(response_dictionary["lighter found response"].format(first_name.capitalize(), 
                                                                    boarding_pass_results[0].fields['Flight Number'].value, 
                                                                    boarding_pass_results[0].fields['Boarding Time.'].value, 
                                                                    boarding_pass_results[0].fields['Departure Location'].value, 
                                                                    boarding_pass_results[0].fields['Arrival Location'].value, 
                                                                    boarding_pass_results[0].fields['Seat Allocation'].value))
    
    
    

    flight_manifest.to_csv("flight_manifest.csv", index=False)

else:
    print("Sorry this name does not appear on the flight bookings list!")
