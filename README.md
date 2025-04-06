# 📘 Facial Expression Recognition System

## 🧠 Project Introduction

This project is a facial expression recognition system based on deep learning, combining ResNet, FPN, and Transformer architectures to achieve high-accuracy emotion recognition. The system provides an interactive web interface supporting image upload analysis, real-time webcam recognition, and 3D neural network visualization. It's suitable for emotion analysis, human-computer interaction, and other application scenarios.

## 🗂️ Project Structure

```
├── fer2013_ResNet-FPN-Transformer/   # Best performing hybrid model
│   ├── final_complete_model/         # Saved model
│   ├── confusion_matrix.png          # Confusion matrix
│   └── training_curves.png           # Training curves
├── fer2013_ResNet-ViT/               # ResNet-ViT hybrid model
│   ├── final_model_no_fpn/           # Saved model
│   └── confusion_matrix_no_fpn.png   # Confusion matrix
├── fer2013_ResNet50/                 # Baseline ResNet50 model
│   ├── 20250323-215041/              # Training results
│   │   ├── tensorboard_logs/         # TensorBoard logs
│   │   ├── best_model.h5             # Best model
│   │   ├── confusion_matrix.png      # Confusion matrix
│   │   └── training_curves.png       # Training curves
│   └── resnet50.py                   # Model definition script
├── web/                              # Web application
│   ├── css/                          # Stylesheets
│   │   ├── featureMap.css            # Feature map visualization styles
│   │   ├── networkVis.css            # Network visualization styles
│   │   └── style.css                 # Main stylesheet
│   ├── js/                           # JavaScript files
│   │   ├── featureMap.js             # Feature map visualization
│   │   ├── main.js                   # Main application logic
│   │   └── resnetVis.js              # 3D network visualization
│   ├── tfjs_model/                   # Converted TensorFlow.js model
│   ├── tfjs-tfjs-v4.22.0/            # TensorFlow.js library
│   └── index.html                    # Main HTML file
├── LICENSE                           # License file
└── README.md                         # Project documentation
```

## 🧰 Technology Stack

| Component       | Technologies                         |
|-----------------|--------------------------------------|
| Model Training  | Python, TensorFlow/Keras             |
| Dataset         | FER-2013                             |
| Frontend UI     | HTML, CSS, JavaScript                |
| Real-time Recognition | TensorFlow.js, WebRTC          |
| 3D Visualization| Three.js                             |
| Chart Display   | Chart.js                             |

## 😃 Supported Expression Categories

| Label | Expression Name |
|-------|----------------|
| 0     | Angry          |
| 1     | Disgust        |
| 2     | Fear           |
| 3     | Happy          |
| 4     | Sad            |
| 5     | Surprise       |
| 6     | Neutral        |

## 📊 Model Performance

| Model                      | Accuracy | Precision | Recall | F1   | Training Time |
|---------------------------|----------|-----------|--------|------|---------------|
| ResNet50                  | 78.94%   | 0.78      | 0.79   | 0.78 | 25 minutes    |
| ResNet-ViT                | 80.45%   | 0.75      | 0.67   | 0.70 | 35 minutes    |
| ResNet+FPN+Transformer    | 80.65%   | 0.78      | 0.68   | 0.72 | 42 minutes    |

The ResNet+FPN+Transformer hybrid model combines:
- ResNet backbone for local feature extraction
- Feature Pyramid Network (FPN) for multi-scale feature representation
- Transformer encoder for capturing global relationships

## 🚀 Quick Start

### 1️⃣ Clone the Project

```bash
git clone https://github.com/Keyuuuuuu/AI-in-web-applicaton.git
cd AI-in-web-applicaton
```

### 2️⃣ Run the Web Application

#### Method 1: Direct Opening
Open the index.html file in the web directory

#### Method 2: Using a Local Server
```bash
cd web
# Python 3
python -m http.server 8000
# or Python 2
python -m SimpleHTTPServer 8000
```

Then open `http://localhost:8000` in your browser

### 3️⃣ Using the Application

1. **Image Mode**:
   - Click "Choose File" to upload an image
   - View the predicted emotion, confidence scores, and feature map visualization

2. **Camera Mode**:
   - Click "Start Camera" to activate your webcam
   - View real-time emotion predictions as your expressions change

3. **Network Structure**:
   - Explore the 3D visualization of the neural network
   - Interact with nodes to see details about each layer

## ✨ Feature Visualization

The system provides unique neural network visualization capabilities:
- Feature map animations: Show how the model "sees" facial expressions at different layers
- 3D network visualization: Displays the model structure in an interactive three-dimensional way
- Confidence charts: Presents prediction probabilities for emotion categories through dynamic charts


## 🌐 Real-time Recognition Demo

The web application uses WebRTC for camera access and TensorFlow.js for model inference. The system adopts a client-side processing strategy, ensuring all facial image analysis is done in the user's browser, enhancing privacy protection.


## 📝 Acknowledgments

- Dr. Kovásznai Gergely (Supervisor)
- Eszterházy Károly Catholic University, Faculty of Informatics
- The creators of the FER-2013 dataset

## 📞 Contact

Hu Ming - mitntghu@gmail.com
Project Link: https://github.com/Keyuuuuuu/AI-in-web-applicaton
