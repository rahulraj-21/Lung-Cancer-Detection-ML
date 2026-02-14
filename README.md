# Lung Cancer Detection using Machine Learning & Computer Vision

## Project Overview
This project focuses on early lung cancer detection using deep learning techniques applied to lung CT scan images. The system classifies images as Benign or Malignant using a trained CNN-based fusion model.

## Features
- Image-based lung cancer prediction
- Deep Learning based classification
- Accuracy & Loss Visualization
- Confusion Matrix Evaluation
- Flask-based Web Interface

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Flask
- NumPy, Pandas
- HTML/CSS (Frontend)

## Project Structure
lung-cancer-detection/
│── app.py  
│── training.py  
│── testing.py  
│── requirements.txt  
│── templates/  
│── static/  
│── models/  

## How to Run
1. Install dependencies:
pip install -r requirements.txt

2. Run application:
python app.py

## Dataset
Publicly available lung CT scan dataset used for training.

## Output
Model predicts whether the input CT image is:
- Benign
- Malignant
