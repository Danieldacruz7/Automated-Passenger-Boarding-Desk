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

list_of_id_results = []
test_images = [file for file in glob.glob("./data/digital_id_template/Test-Images/ca-dl-*.png")]
for image_path in test_images:
        with open(image_path, "rb") as test_data:
                results = form_recognizer_client.begin_recognize_identity_documents(test_data, content_type="image/png")
        list_of_id_results.append(results.result())

list_of_ids = []
for i in list_of_id_results:
    dict_results = {}
    for key, value in (i[0].fields).items():
        dict_results[key] = value.value
    list_of_ids.append(dict_results)

training_images_url = "https://udacitystorageaccount111.blob.core.windows.net/custom-form?sp=racwdl&st=2022-11-07T02:19:28Z&se=2022-11-14T11:19:28Z&spr=https&sv=2021-06-08&sr=c&sig=WAIHYrZhno1sSfIWH2kFY2G35nxwQHDXGHqIHPmKj8g%3D"
training_process = form_training_client.begin_training(training_images_url, use_training_labels=True)
custom_model = training_process.result()

custom_model_info = form_training_client.get_custom_model(model_id=custom_model.model_id)
print("Model ID: {}".format(custom_model_info.model_id))
print("Status: {}".format(custom_model_info.status))
print("Training started on: {}".format(custom_model_info.training_started_on))
print("Training completed on: {}".format(custom_model_info.training_completed_on))

list_of_boarding_pass_results = []

test_images = [file for file in glob.glob("./data/boarding_pass_template/Test-Images/*.pdf")]
for image_path in test_images:
        print(image_path)
        with open(image_path, "rb") as test_data:
                results = form_recognizer_client.begin_recognize_custom_forms(model_id=custom_model_info.model_id, form = test_data, content_type='application/pdf')
        list_of_boarding_pass_results.append(results.result())

boarding_pass_results = []
for i in list_of_boarding_pass_results:
    dict_results = {}
    for key, value in (i[0].fields).items():
        dict_results[key] = value.value
    boarding_pass_results.append(dict_results)

# Upload to Video Analzyer from local disk
uploaded_video_id = video_analysis.upload_to_video_indexer(
   input_filename=r'data\digital-video-sample\Azure-video-submission.mp4',
   video_name='daniel-da-cruz-boarding-pass',  # unique identifier for video in Video Indexer platform
   video_language='English'
)
time.sleep(300)
info = video_analysis.get_video_info(uploaded_video_id, video_language='English')

images = []
img_raw = []
img_strs = []
thumbnails = []
for each_thumb in info['videos'][0]['insights']['faces'][0]['thumbnails']:
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

thumbnail_directory = "./data/ai-generated-thumbnails/"
i = 1
for img in images:
    img.save(thumbnail_directory + 'human-face' + str(i) + '.jpg')
    i= i+ 1
print("Thumbnails saved to {}".format(thumbnail_directory))

img_code = video_analysis.get_thumbnail_from_video_indexer(uploaded_video_id,  thumbnails[0])

PERSON_GROUP_ID = str(uuid.uuid4())
person_group_name = 'daniel'
face_client = FaceClient(CONFIG['FACIAL_RECOGNITION_ENDPOINT'], CognitiveServicesCredentials(CONFIG['FACIAL_RECOGNITION_KEY']))

def build_person_group(client, person_group_id, pgp_name, directory):
    for file in glob.glob(directory + '*.jpg'):
        print(file)
    human_face_images = [file for file in glob.glob('*.jpg') if file.startswith(directory + "human-face")]
    print(human_face_images)
    print('Create and build a person group...')
    # Create empty Person Group. Person Group ID must be lower case, alphanumeric, and/or with '-', '_'.
    print('Person group ID:', person_group_id)
    client.person_group.create(person_group_id = person_group_id, name=person_group_id)

    # Create a person group person.
    human_person = client.person_group_person.create(person_group_id, pgp_name)
    # Find all jpeg human images in working directory.
    human_face_images = [file for file in glob.glob(directory + '*.jpg')]
    # Add images to a Person object
    for image_p in human_face_images:
        with open(image_p, 'rb') as w:
            client.person_group_person.add_face_from_stream(person_group_id, human_person.person_id, w)

    # Train the person group, after a Person object with many images were added to it.
    client.person_group.train(person_group_id)

    # Wait for training to finish.
    while (True):
        training_status = client.person_group.get_training_status(person_group_id)
        print("Training status: {}.".format(training_status.status))
        if (training_status.status is TrainingStatusType.succeeded):
            break
        elif (training_status.status is TrainingStatusType.failed):
            client.person_group.delete(person_group_id=PERSON_GROUP_ID)
            sys.exit('Training the person group has failed.')
        time.sleep(5)
        
build_person_group(face_client, PERSON_GROUP_ID, person_group_name, "./data/ai-generated-thumbnails/")

def detect_faces(client, query_images_list):
    print('Detecting faces in query images list...')

    face_ids = {} # Keep track of the image ID and the related image in a dictionary
    for image_name in query_images_list:
        image = open(image_name, 'rb') # BufferedReader
        print("Opening image: ", image.name)
        time.sleep(5)

        # Detect the faces in the query images list one at a time, returns list[DetectedFace]
        faces = client.face.detect_with_stream(image)  

        # Add all detected face IDs to a list
        for face in faces:
            print('Face ID', face.face_id, 'found in image', os.path.splitext(image.name)[0]+'.jpg')
            # Add the ID to a dictionary with image name as a key.
            # This assumes there is only one face per image (since you can't have duplicate keys)
            face_ids[image.name] = face.face_id

    return face_ids

test_images = [file for file in glob.glob('./data/ai-generated-thumbnails/*.jpg')]
ids = detect_faces(face_client, test_images)

def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height
    
    return ((left, top), (right, bottom))

def drawFaceRectangles(source_file, detected_face_object) :
    # Download the image from the url
    response = requests.get(source_file)
    img = Image.open(BytesIO(response.content))
    # Draw a red box around every detected faces
    draw = ImageDraw.Draw(img)
    for face in detected_face_object:
        draw.rectangle(getRectangle(face), outline='red', width = 2)
    return img

image_url_daniel_da_cruz = "https://udacitystorageaccount111.blob.core.windows.net/digital-id/ca-dl-daniel-da-cruz.png?sp=r&st=2022-11-08T12:22:34Z&se=2022-11-15T20:22:34Z&spr=https&sv=2021-06-08&sr=b&sig=VJwPE5wDWXI6gU3WEMcdFwKSjrVTfa%2FsTl3L3jZnL5c%3D"
image_url_helena_da_cruz = "https://udacitystorageaccount111.blob.core.windows.net/digital-id/ca-dl-helena-da-cruz.png?sp=r&st=2022-11-08T12:23:27Z&se=2022-11-15T20:23:27Z&spr=https&sv=2021-06-08&sr=b&sig=BlN0Le%2BcLXKxolJ6fj2DH3sIhNKi10d23DtOxlVqox0%3D"
image_url_john_doe = "https://udacitystorageaccount111.blob.core.windows.net/digital-id/ca-dl-john-doe.png?sp=r&st=2022-11-08T12:24:46Z&se=2022-11-15T20:24:46Z&spr=https&sv=2021-06-08&sr=b&sig=%2FIgg7RM3H2DBKjsbsf3S1t%2BX0iA8pwxKOOlbKuXFEDM%3D"
image_url_mark_musk = "https://udacitystorageaccount111.blob.core.windows.net/digital-id/ca-dl-mark-musk.png?sp=r&st=2022-11-08T12:25:36Z&se=2022-11-15T20:25:36Z&spr=https&sv=2021-06-08&sr=b&sig=%2F81fcgnXTo77uoiXPQSUEDlpZ7ZN8rlKPScFPliPbbg%3D"
image_url_noah_taleb = "https://udacitystorageaccount111.blob.core.windows.net/digital-id/ca-dl-noah-taleb.png?sp=r&st=2022-11-08T12:26:04Z&se=2022-11-15T20:26:04Z&spr=https&sv=2021-06-08&sr=b&sig=yhBO8WyBRxwWI62nvfz%2FOM1eP34fGtWEPEW%2FbZ5eyoE%3D"

def detect_face_with_attributes_02_from_any_url(selected_image_url):
    detected_faces = face_client.face.detect_with_url(url=selected_image_url, 
                                                     return_face_attributes=[
                    'age',
                    'gender',
                    'headPose',
                    'smile',
                    'facialHair',
                    'glasses',
                    'emotion',
                    'hair',
                    'makeup',
                    'occlusion',
                    'accessories',
                    'blur',
                    'exposure',
                    'noise'
                ])
    if not detected_faces:
        raise Exception('No face detected from image {}'.format(selected_image_url))        
    print('Total face(s) detected  from {}'.format(str(len(detected_faces))))
    return detected_faces
