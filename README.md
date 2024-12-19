# Face Recognition Attendance System

## Installation Options

### Option 1: Quick Setup (Recommended)
This will create an exact copy of the development environment:

1. **Create and activate a Conda environment:**

conda create -n face_recognition python=3.9
conda activate face_recognition

for nvidia 


2. **Install CUDA Toolkit (if using GPU):**
- Download and install CUDA Toolkit 11.8 from: https://developer.nvidia.com/cuda-11-8-0-download-archive
- Download and install cuDNN for CUDA 11.8 from: https://developer.nvidia.com/cudnn

3. **Install dependencies:**
Install PyTorch (required for insightface)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
Install other dependencies
pip install -r requirements.txt

FACE_RECOG/
├── assets/
│ ├── student_images/
│ │ ├── 001/
│ │ │ ├── front.jpg
│ │ │ ├── left.jpg
│ │ │ └── right.jpg
│ │ └── 002/
│ │ └── ...
│ └── face_encodings/
├── main.py
└── requirements.txt


5. **Usage:**
- Place student images in folders under assets/student_images/
- Each student should have their own folder named with their ID
- Run the program:


## Troubleshooting

1. **If you don't have a GPU or have GPU issues:**
- Change DEVICE = "gpu" to DEVICE = "cpu" in main.py
- Install onnxruntime instead of onnxruntime-gpu:


if onx issue due to not having gpu
try this 
pip uninstall onnxruntime-gpu
pip install onnxruntime


2. **Common Issues:**
- If you get "DLL load failed" errors on Windows, install Visual C++ Redistributable:
  - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

3. **Performance Tips:**
- GPU mode is significantly faster for training and recognition
- CPU mode works but will be slower, especially during training
- Use well-lit, clear photos for best recognition results

## System Requirements

- Python 3.9 or later
- 4GB RAM minimum (8GB recommended)
- NVIDIA GPU with CUDA support (optional, but recommended)
- Webcam

# CPU-Only Setup Instructions

1. **Create and activate Conda environment:**

conda create -n face_recognition python=3.9
conda activate face_recognition

for nvidia 


2. **Install CUDA Toolkit (if using GPU):**
- Download and install CUDA Toolkit 11.8 from: https://developer.nvidia.com/cuda-11-8-0-download-archive
- Download and install cuDNN for CUDA 11.8 from: https://developer.nvidia.com/cudnn

3. **Install dependencies:**
Install PyTorch (required for insightface)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
Install other dependencies
pip install -r requirements.txt

FACE_RECOG/
├── assets/
│ ├── student_images/
│ │ ├── 001/
│ │ │ ├── front.jpg
│ │ │ ├── left.jpg
│ │ │ └── right.jpg
│ │ └── 002/
│ │ └── ...
│ └── face_encodings/
├── main.py
└── requirements.txt


5. **Usage:**
- Place student images in folders under assets/student_images/
- Each student should have their own folder named with their ID
- Run the program:


## Troubleshooting

1. **If you don't have a GPU or have GPU issues:**
- Change DEVICE = "gpu" to DEVICE = "cpu" in main.py
- Install onnxruntime instead of onnxruntime-gpu:


if onx issue due to not having gpu
try this 
pip uninstall onnxruntime-gpu
pip install onnxruntime


2. **Common Issues:**
- If you get "DLL load failed" errors on Windows, install Visual C++ Redistributable:
  - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

3. **Performance Tips:**
- GPU mode is significantly faster for training and recognition
- CPU mode works but will be slower, especially during training
- Use well-lit, clear photos for best recognition results

## System Requirements

- Python 3.9 or later
- 4GB RAM minimum (8GB recommended)
- NVIDIA GPU with CUDA support (optional, but recommended)
- Webcam
