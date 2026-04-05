# SecureWatch — AI Industrial Surveillance

SecureWatch is a real-time industrial surveillance application powered by computer vision. It uses YOLOv8 for human detection and provides an interactive dashboard for monitoring safety, detecting intrusions, and analyzing site activity.

## Overview

The system is designed to enhance safety and security in industrial environments. It combines real-time detection with a modern user interface built using Streamlit. The application allows users to monitor camera feeds, detect human presence, and visualize alerts and performance metrics.

## Features

- Real-time human head detection using YOLOv8  
- Interactive dashboard with alerts and key performance indicators  
- Image upload and webcam-based detection  
- Adjustable detection parameters (confidence threshold, IOU, head ratio)  
- Incident timeline and camera status monitoring  
- Custom user interface with responsive layout  

## Technologies

- Python  
- Streamlit  
- OpenCV  
- Ultralytics YOLOv8  
- NumPy  
- Pandas  
- Pillow  

## Project Structure 
modl.py Main Streamlit application
README.md Project documentation
requirements.txt Project dependencies

## Installation
1. Clone the repository:
2. git clone https://github.com/your-username/your-repository.git
cd your-repository

2. (Optional) Create a virtual environment:
python -m venv venv
venv\Scripts\activate (Windows)
source venv/bin/activate (Linux/Mac)

3. Install dependencies:
pip install -r requirements.txt

## Usage
Run the application using Streamlit:
streamlit run modl.py
Then open your browser at:
http://localhost:8501


## System Description

The application uses YOLOv8 to detect persons in images. For each detected person, a region corresponding to the head is estimated based on a configurable ratio. The system overlays bounding boxes and displays real-time information such as detection count and confidence levels.

The interface is divided into multiple sections:
- Home: presentation of the system and its capabilities  
- Dashboard: real-time monitoring with alerts and statistics  
- Detection: image and webcam-based analysis  
- Contact: project and team information  

## Team

- Doha Zilaoui — Lead Developer  
- Soukaina Hachmoud — Computer Vision Engineer  
- Sara Fadil — Backend and Alerts Engineer  

## Future Work

- Integration of live video streams (RTSP cameras)  
- Personal protective equipment detection  
- Fall detection using pose estimation  
- Real-time alert notifications (SMS, email)  
- Deployment on edge computing devices  

## License

This project is developed for academic and research purposes.
