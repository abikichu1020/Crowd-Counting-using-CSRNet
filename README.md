# Crowd Counting using CSRNet

## Project Title
Crowd Counting using CSRNet (Convolutional Neural Network)

## Repository
Crowd-Counting-using-CSRNet.git

## Objective
The objective of this project is to estimate the number of people in a crowded image using deep learning. The system uses CSRNet, a convolutional neural network designed for crowd counting in highly dense scenes where traditional object detection fails.

## Description
Crowd counting in dense environments is challenging due to occlusion, scale variation, and perspective distortion. This project implements a deep learning–based approach using CSRNet to predict a density map from an input image. The total crowd count is obtained by integrating over the predicted density map.

The project also includes a simple web-based interface to upload images and visualize predictions, making it suitable for demonstrations and academic use.

## Project Structure
- model.py : Defines and loads the CSRNet model architecture
- predict.py : Performs crowd counting inference on images
- predict1.py : Alternative or extended prediction logic
- index.html : Web interface for image upload and result display
- static/ : Static assets for the web interface
- uploads/ : Stores uploaded images
- crowded.jpg : Sample crowded image for testing
- yolov5su.pt : Pretrained model file (if used for auxiliary tasks)
- requirements.txt : Python dependencies
- __pycache__/ : Python cache files

## Technologies Used
- Python
- Deep Learning (CNN)
- CSRNet architecture
- PyTorch
- HTML / CSS (basic frontend)
- Flask (for web app integration)

## Model Used
CSRNet (Congested Scene Recognition Network)

Key characteristics:
- Fully convolutional network
- Uses dilated convolutions
- Generates density maps instead of bounding boxes
- Effective for high-density crowd scenes

## Workflow
1. User uploads a crowded image
2. Image is preprocessed
3. CSRNet predicts a density map
4. Density map is summed to obtain crowd count
5. Result is displayed to the user

## Libraries Used
- torch
- torchvision
- numpy
- opencv-python
- pillow
- flask

## Features
- Accurate crowd estimation in dense scenes
- Density map–based counting
- Simple web interface
- Pretrained model support
- Suitable for academic and demo purposes

## Use Cases
- Crowd monitoring
- Public safety analysis
- Event management
- Smart city applications
- Surveillance analytics

## Advantages
- Works well in highly crowded images
- Avoids individual person detection
- Robust to occlusion and scale variation
- Faster inference compared to detection-based methods

## Limitations
- Requires pretrained model weights
- Performance depends on dataset quality
- Not suitable for sparse crowd detection
- No real-time video processing implemented

## How to Run
1. Install Python (3.x recommended)
2. Install dependencies:
   pip install -r requirements.txt
3. Ensure pretrained CSRNet model is available
4. Run the prediction script:
   python predict.py
   or
   python predict1.py
5. Open index.html or run Flask app (if configured)
6. Upload an image to get crowd count

## Output
- Predicted crowd count
- Density map visualization (if enabled)
- Processed image output

## Conclusion
This project demonstrates an effective deep learning approach for crowd counting using CSRNet. By leveraging density map estimation instead of object detection, it provides accurate results in highly congested scenes, making it suitable for real-world surveillance and smart city applications.

## Author
Developed as part of a computer vision / deep learning academic project.
