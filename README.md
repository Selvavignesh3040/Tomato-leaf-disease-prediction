Tomato Leaf Disease Detection
Introduction
Tomato Leaf Disease Detection is a deep learning-based project aimed at identifying diseases in tomato leaves using Convolutional Neural Networks (CNNs). The model classifies leaves into three categories: Healthy, Early Blight, and Late Blight. This tool helps farmers and agricultural professionals diagnose plant diseases early, leading to timely interventions and preventing crop loss.

Features
Automated disease classification for tomato leaves.
Fast, accurate, and cost-effective compared to manual detection methods.
Modular design for easy scalability to other crops or diseases.
Standalone prediction script for real-time deployment.
Dataset
This project uses a custom dataset containing images of tomato leaves categorized into three classes:

Healthy: Healthy tomato leaves.
Early Blight: Leaves affected by Early Blight disease.
Late Blight: Leaves affected by Late Blight disease.
You can download the dataset from the following link:

Dataset Link - Google Drive: https://drive.google.com/drive/folders/1M1Rtd4btIJtG24QZjSANmBAdmmnzjzz-?usp=sharing

Dataset Structure:
bash
Copy code
dataset/
├── train/
│   ├── Healthy/
│   ├── Early_Blight/
│   └── Late_Blight/
└── test/
    ├── Healthy/
    ├── Early_Blight/
    └── Late_Blight/
Setup and Installation
1. Clone the repository
First, clone the repository to your local machine:

bash
Copy code
git clone https://github.com/Selvavignesh3040/tomato-leaf-disease-detection.git
cd tomato-leaf-disease-detection
2. Set up a virtual environment
Create a virtual environment to keep dependencies isolated:

bash
Copy code
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
3. Install required dependencies
Install all necessary Python packages listed in requirements.txt:

Copy code
pip install -r requirements.txt
Usage
Training the Model
To train the model on the dataset, run the following command:

bash
Copy code
python src/train.py
The script will train the model using the train/ images and save the trained model as tomato_model.h5 in the models/ directory.

Predicting with the Model
To make predictions with the trained model, use the predict.py script. Provide the path to a leaf image as input:

css
Copy code
python src/predict.py --image path/to/leaf_image.jpg
The script will output the predicted disease (Healthy, Early Blight, or Late Blight) and the confidence level.

Results
The model achieves a high accuracy in classifying tomato leaf diseases, providing a reliable tool for agricultural professionals to detect early-stage diseases.

Example prediction output:

less
Copy code
Input Image: leaf_image.jpg
Prediction: Early Blight (Confidence: 92%)
Future Work
Extend detection to other crops such as potato, bell pepper, etc.
Real-time disease detection using IoT-based systems for field monitoring.
Improve model performance using advanced techniques like Transfer Learning.
License
This project is licensed under the MIT License. See the LICENSE file for details.
