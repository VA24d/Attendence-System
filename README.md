# Face Recognition Attendance System

## Overview

The Face Recognition Attendance System is a Python-based application that uses face recognition technology to manage attendance. It leverages machine learning models for face detection and recognition, and includes features for spoof detection to ensure the authenticity of the detected faces.

## Features

- **Face Registration**: Capture and register new faces into the system.
- **Face Recognition**: Recognize faces from live video feed or uploaded images.
- **Spoof Detection**: Detect and prevent spoofing attempts using advanced algorithms.
- **Device Configuration**: Support for both CPU and GPU processing.
- **Theme Toggle**: Switch between light and dark themes.
- **Fullscreen Mode**: Toggle fullscreen mode for the application.
- **Student Management**: Update and manage the list of registered students.

## Installation

### Option 1: Quick Setup with Conda (Recommended)

This will create an exact copy of the development environment.

1. **Create and activate a Conda environment:**

    ```sh
    conda create -n face_recognition python=3.9
    conda activate face_recognition
    ```

### Option 2: Manual Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/face_recognition_attendance_system.git
    cd face_recognition_attendance_system
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Place student images in folders under [student_images]

    - Each student should have their own folder named with their ID.
    - Example structure:

    ```
    assets/
    ├── student_images/
    │   ├── 001/
    │   │   ├── front.jpg
    │   │   ├── left.jpg
    │   │   └── right.jpg
    │   └── 002/
    │       └── ...
    └── face_encodings/
    ```

2. **Run the program:**

    ```sh
    python FaceRec.py
    ```

3. **Use the GUI to register new faces, recognize faces, and manage the system.**

## Troubleshooting

1. **Common Issues:**

    - If you get "DLL load failed" errors on Windows, install Visual C++ Redistributable:
      - Download from: [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

2. **Performance Tips:**

    - MPS mode is faster for training and recognition.
    - CPU mode works but will be slower, especially during training.
    - Use well-lit, clear photos for best recognition results.

## System Requirements

- Python 3.9 or later
- 8GB RAM minimum
- Mac OS X or Linux (Windows is supported but not recommended)
- Webcam

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [InsightFace](https://github.com/deepinsight/insightface) for face detection and recognition.
- [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) for spoof detection.
