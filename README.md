# deepfake-detection

# Dataset
The Dataset we used is celebDF which you can find [here](https://github.com/yuezunli/celeb-deepfakeforensics) .For computational efficiency we only used around 1168 videos.

# Preprocessing 

1. Frame Extraction: Split the videos into individual frames.
2. Face Detection and Cropping: Used OpenCV's Cascade Classifier to detect and crop faces.
3. Video Reconstruction: Stitched the processed frames back into videos.

# Model Training 
Refer to 'Model_training_DF.ipynb' for detailed instructions and code on training the model.

# Deployment :
Use 'app.py' to deploy your trained model as a user-friendly web application. This script sets up a web interface where users can upload videos for deepfake detection.

For further details on usage and configuration, please check the provided notebooks and scripts in the repository.
