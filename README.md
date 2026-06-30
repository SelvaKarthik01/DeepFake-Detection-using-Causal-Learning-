# DeepFake Detection using Causal Learning

## �📋 Overview

This repository contains a comprehensive machine learning pipeline for **detecting deepfakes and manipulated videos** using causal learning principles and facial landmark analysis. The project uses the **Face Forensics++ dataset** to train deep learning models that can classify videos into 7 categories: original videos and 6 types of deepfake manipulations.

### Dataset
- **Face Forensics++ Dataset** → [Kaggle Link](https://www.kaggle.com/datasets/xdxd003/ff-c23)

---

## 🎯 Project Objectives

1. **Golden Frame Extraction**: Identify and extract high-quality frames from videos using facial symmetry and sharpness metrics
2. **Facial Landmark Jitter Analysis**: Compute landmark jitter (motion between frames) as a feature for deepfake detection
3. **Face Cropping & Preprocessing**: Efficiently preprocess extracted frames for model training
4. **Multi-Class Classification**: Classify videos into 7 categories with high accuracy
5. **Causal Learning Approach**: Apply causal inference principles to improve model robustness and interpretability

---

## 🏗️ Project Architecture

<img width="872" height="597" alt="image" src="https://github.com/user-attachments/assets/e4914483-f941-4e80-a94b-75cf26396532" />


### 1. **Golden Frames Extraction** (`Golden Frames Extracter/`)
   - **File**: `GoldenFramestester.py`
   - **Purpose**: Extract the highest quality frames from videos
   - **Methodology**:
     - Uses **MediaPipe FaceMesh** for facial landmark detection
     - Implements multi-tier filtering:
       - **Sharpness Check**: Laplacian blur detection to identify clear frames
       - **Frontal Pose Check**: Facial symmetry analysis using eye-to-nose distances
       - **Confidence Scoring**: MediaPipe detection confidence filtering
     - Returns frames where the face is most frontal and sharp

### 2. **Golden Frames Landmark Jitter** (`Golden Frames Landmark Jitter/`)
   - **Main File**: `GoldenFrames.py`
   - **Purpose**: Analyze facial landmark movements (jitter) between frames
   - **Key Components**:
     - **FaceCropMeshDetector Class**: 
       - Step 1: MediaPipe FaceDetection for initial bounding box
       - Step 2: Crop the detected ROI
       - Step 3: Apply FaceMesh to cropped region for precise landmarks
       - Returns pixel coordinates in original image space
     - **Jitter Computation**: Computes median euclidean distance between landmark positions across consecutive frames
     - **Key Landmarks**: Eyes, nose, and mouth regions (critical for detecting manipulation artifacts)

### 3. **Dataset Preprocessing** 
   - **Jitter Dataset Preprocessing.ipynb**: Processes raw videos into jitter feature datasets
   - **Jitter Facecrop Dataset Preprocessing.ipynb**: Combines jitter analysis with face cropping
   - **Processing Steps**:
     - Extract landmarks at fixed intervals (configurable FPS)
     - Compute jitter scores for temporal consistency
     - Filter frames based on landmark detection confidence
     - Generate balanced training/validation splits

### 4. **Main Classification Model** (`final code.ipynb`)
   - **Multi-Class Classification**: 7-class problem
     - `DeepFakeDetection` (synthetic detection artifact)
     - `Deepfakes` (deepfake generation method)
     - `Face2Face` (expression manipulation)
     - `FaceShifter` (identity swap)
     - `FaceSwap` (identity swap)
     - `NeuralTextures` (texture synthesis)
     - `original` (authentic/real videos)
   - **Model Configuration**:
     - Batch Size: 32
     - Epochs: 15
     - Learning Rate: 2e-4
     - Image Size: 224×224
     - Uses Pre-trained PyTorch Models (ResNet, etc.)
   - **Evaluation Metrics**:
     - Accuracy & F1-Score
     - Confusion Matrix
     - Classification Report
     - ROC-AUC Score
     - Top-K Accuracy
    
   <img width="840" height="832" alt="image" src="https://github.com/user-attachments/assets/c2898db5-dc6b-43d0-930d-ecb937005cf4" />


---

## 🔧 Tech Stack

### Core Libraries
- **PyTorch**: Deep learning framework
- **TorchVision**: Pre-trained models and image transforms
- **MediaPipe**: Facial detection and landmark extraction
- **OpenCV**: Video processing and image manipulation
- **Scikit-Learn**: Model evaluation and metrics
- **NumPy & Pandas**: Data manipulation
- **Matplotlib**: Visualization

### Requirements
```
opencv-python==4.9.0.80
mediapipe==0.10.14
numpy==1.26.4
pandas==2.2.2
tqdm==4.66.4
torch (latest GPU-compatible version)
torchvision (compatible with PyTorch version)
scikit-learn
matplotlib
```

---

## 📁 Repository Structure

```
DeepFake-Detection-using-Causal-Learning/
├── README.md                                    # This file
├── final code.ipynb                             # Main classification model training & evaluation
├── Jitter_Dataset_Preprocessing.ipynb           # Jitter feature extraction pipeline
├── Jitter_Facecrop_Dataset_Preprocessing.ipynb  # Combined jitter + face cropping pipeline
├── Base Paper.pdf                               # Research paper foundation
├── Major Project Abstract.docx/.pdf             # Project abstract
│
├── Golden Frames Extracter/
│   └── GoldenFramestester.py                   # Extract high-quality frames from videos
│
├── Golden Frames Landmark Jitter/
│   ├── GoldenFrames.py                         # Landmark jitter computation with face detection
│   ├── requirements.txt                        # Python dependencies
│   └── README.md                               # Module-specific documentation
│
└── Literature Review/
    ├── README.md                               # Literature review summary
    ├── DeepFake Literature Review Excel Template.xlsx
    └── Reference Literature Review Excel Sheet.xlsx
```

---

## 🚀 Workflow & Usage

### Step 1: Install Dependencies
```bash
pip install -r "Golden Frames Landmark Jitter/requirements.txt"
pip install torch torchvision scikit-learn matplotlib
```

### Step 2: Extract Golden Frames
Run the golden frame extraction to identify high-quality frames from video files:
```python
from Golden Frames Extracter.GoldenFramestester import extract_golden_frames

video_path = "path/to/video.mp4"
golden_frames = extract_golden_frames(video_path)
# Returns: list of frame indices with best quality metrics
```

### Step 3: Compute Facial Landmark Jitter
Process extracted frames and compute jitter features:
```python
# Run the Jitter_Dataset_Preprocessing.ipynb notebook
# This will:
# 1. Extract facial landmarks from frames
# 2. Compute jitter (motion between frames)
# 3. Save features to structured dataset
```

### Step 4: Preprocess Face Crops
If using face-cropped datasets, run:
```python
# Run Jitter_Facecrop_Dataset_Preprocessing.ipynb
# This combines face cropping with jitter analysis
```

### Step 5: Train Classification Model
Execute the main training notebook:
```python
# Run final code.ipynb
# This will:
# 1. Load preprocessed data
# 2. Initialize model (ResNet/other architecture)
# 3. Train for 15 epochs
# 4. Generate evaluation metrics and visualizations
```

---

## 🧠 Causal Learning Approach

This project integrates **causal learning principles** to improve deepfake detection:

- **Causal Features**: Uses facial landmarks (direct causes of motion) rather than raw pixels
- **Temporal Causality**: Analyzes how manipulation artifacts propagate through video sequences
- **Jitter as Causal Signal**: Landmark jitter acts as a causal indicator—genuine faces have natural motion patterns, while deepfakes exhibit unnatural movement artifacts
- **Robustness**: Causal features are more resistant to adversarial attacks and generalize better across different deepfake generation methods

---

## 📊 Key Features of the Pipeline

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Golden Frame Selection** | Extracts frames where face is most frontal & sharp | Improves model training quality |
| **Facial Landmark Jitter** | Temporal motion analysis of 9 key facial points | Detects manipulation-induced artifacts |
| **Multi-Tier Filtering** | Sharpness + pose + confidence checks | Eliminates low-quality training samples |
| **7-Class Classification** | Real + 6 manipulation methods | Identifies specific attack type |
| **GPU Acceleration** | Full CUDA support | Fast training on modern GPUs |
| **Comprehensive Metrics** | F1, Confusion Matrix, ROC-AUC, Top-K Accuracy | Detailed performance analysis |

---

## 🔍 Model Performance

The model is trained to achieve:
- **High Accuracy** on Face Forensics++ dataset (7-class classification)
- **Balanced Performance** across all deepfake manipulation types
- **Robust Detection** through causal feature analysis
- **Generalization** to unseen manipulation techniques

---

## 📚 Literature & Research Foundation

This project is based on peer-reviewed research in:
- **Deepfake Detection**: Analyzing manipulation artifacts and temporal consistency
- **Causal Inference**: Using causal graphs to identify root causes of face manipulation
- **Facial Analysis**: MediaPipe and facial landmark-based features
- **Face Forensics++**: Benchmark dataset for forensic analysis

See `Literature Review/` folder for detailed research papers and references.

---

## 🎓 Project Context

- **Type**: Machine Learning / Computer Vision / Digital Forensics
- **Dataset**: Face Forensics++ (C23 subset)
- **Models**: Deep Convolutional Neural Networks with Pre-training
- **Primary Language**: Python 3.x
- **Notebooks**: Jupyter/IPython format for interactive exploration

---

## ⚙️ Key Parameters & Configuration

Located in `final code.ipynb`:
```python
DATA_PATH = "/path/to/face_crops"
BATCH_SIZE = 32                    # Batch size for training
EPOCHS = 15                        # Number of training epochs
LR = 2e-4                         # Learning rate
PATIENCE = 2                       # Early stopping patience
IMG_SIZE = 224                    # Input image size
NUM_CLASSES = 7                   # Number of classification classes
```

---

## 🛠️ Troubleshooting

### Issue: MediaPipe not detecting faces
- **Solution**: Ensure frames have clear facial regions, adjust `min_detection_confidence` (lower = more lenient)

### Issue: Memory issues during training
- **Solution**: Reduce `BATCH_SIZE` or use gradient accumulation

### Issue: Low accuracy on validation set
- **Solution**: Check data preprocessing, ensure balanced class distribution, increase epochs

---

## 📖 How to Extend This Project

1. **Add New Deepfake Detection Methods**: Extend class labels and retrain
2. **Use Different Pre-trained Models**: Modify model architecture in `final code.ipynb`
3. **Temporal Models**: Incorporate 3D CNNs (C3D, SlowFast) for video-level classification
4. **Ensemble Methods**: Combine predictions from multiple models
5. **Explainability**: Use LIME/SHAP to visualize model decisions

---

## 📝 Citation

If you use this project in your research, please cite:
- **Face Forensics++ Dataset**: [Link to paper]
- **MediaPipe**: Google's ML solution for perception
- **Relevant deepfake detection papers** (see Literature Review folder)

---

## 👥 Contributors

- **Selva Karthik** (Repository Owner)
- Contributors welcome! Feel free to submit PRs or issues.

---

## 📄 License

[Add your license information here - e.g., MIT, Apache 2.0, etc.]

---

## 📞 Contact & Support

For questions or issues:
- Open a GitHub Issue
- Check existing issues first
- Provide detailed error messages and environment info

---

## 🔗 Useful Links

- **Face Forensics++ Dataset**: https://www.kaggle.com/datasets/xdxd003/ff-c23
- **MediaPipe Documentation**: https://mediapipe.dev/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Deepfake Research**: [Academic papers in Literature Review folder]

---

**Last Updated**: 2025  
**Status**: Active Development
