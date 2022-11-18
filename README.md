# Automated-Passenger-Boarding-Desk

A system for onboarding passengers using computer vision to extract key information from boarding passes, ID and facial recognition.  

## Table of Contents
1. The Project
2. Installations
3. File Descriptions
4. How To Interact With the Project
5. Licensing, Authors, Acknowledgements

### The Project: 

The goal of the project is to create an automated means of allowing passengers to validate their boarding pass onto their flight without human assistance. The system will be implemented as a self-service kiosk that will use computer vision to extract key information from the boarding pass, digital ID and facial features of each passenger. 

The system will be integrated with Microsoft Azure services. These services include Azure Form Recognizer, Azure Face APIs, Azure Custom Vision, Azure Video Indexer and Azure Blob Storage. The system will use a Python runtime to carry out the process. 

![Architecture Diagram](./Screenshots/Architecture%20Diagram.png)

The project is setup to act as a simulation of the process. The data used in the project is synthetic. The face images were created using generative adverserial networks to avoid privacy concerns. From the site [thispersondoesnotexist.com](https://thispersondoesnotexist.com/), ultra-realistic images of people were created that do not exist. These images were photoshopped to replace the background to that of an airport. 

![ca-dl-mark-musk.png](./data/digital_id_template/Test-Images/ca-dl-mark-musk.png)

After photoshopping the images, a free service on myheritage.com allows one to create realistic moving images using photos of faces. This created 15 second videos of a person moving their face - allowing one to create synthetic videos of people to simulate the onboarding process in an airport.  

![Automated Passenger Kiosk Desk](./Screenshots/Automated%20Passenger%20Kiosk%20Desk.png)

### Installations: 
The following packages will be required for the project: 
1. Pandas
2. Matplotlib
3. Pillow
4. Dotenv
5. Azure core
6. Video Indexer
7. Azure AI Formrecognizer
8. Azure Cognitive Services Vision Computervision

### File Descriptions

- The [Problem Definition & System Design](./1.%20Problem%20Defintion%20%26%20System%20Design/) folder contains the introduction to the project. This includes defining the problem, the dataflow chart and the architecture of the system. 
- The notebooks 2. - 4. contain the workflow for each Azure service: 
    - The [digital text extractor notebook](2.%20digital-text-extractor.ipynb) uses the prebuilt form recognizer model to extract information from the digital ID. A custom-built model was trained to extract information from the boarding passes.
    - The [video analyzer notebook](3.%20video_analyzer.ipynb) uploads video footage to the Video Indexer service and extracts the facial features of the passenger to validate it against the digital ID. Face API service was used to extract facial image data from the digital ID. 
    - The [object detection notebook](4.%20object-detection.ipynb) trains a computer vision model to identify the presence of a prohibited item such as a lighter. This will rule out a passenger from carry the item onboard. 
- The [Data Folder](./data/) contains all the images, boarding pass PDFs to run and test the system.      
- The **main.py** file contains a command-line application for simulating the onboarding process.

### How To Interact With the Project:

To run the main application, run the following in the terminal: 
```
python main.py
``` 
Enter the first name and second name of a passenger on the flight manifest to kickstart the simulation. If the name does not exist, the passenger will be denied entry. This will not be necessary in real life but only to start the simulation process. 

The other notebooks can be viewed to see the workflow of each Azure service and its outputs. 

### Licensing, Authors, Acknowledgements: 

- The facial images used in the dataset were generated using [thispersondoesnotexist.com](https://thispersondoesnotexist.com/). 
- The videos were created using myheritage.com/deep_nostalgia. 
- The license to create this projectcan be found in the license.txt file acknowledging the rights of Udacity. 
