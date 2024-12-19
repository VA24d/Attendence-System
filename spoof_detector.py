import os
import cv2
import numpy as np
import time
import sys
from pathlib import Path
import torch

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent.absolute()

# Add Silent-Face-Anti-Spoofing directory to Python path
SILENT_FACE_DIR = SCRIPT_DIR / "Silent-Face-Anti-Spoofing-master"
if not SILENT_FACE_DIR.exists():
    raise RuntimeError(
        "Silent-Face-Anti-Spoofing-master directory not found! "
        "Please ensure it's in the same directory as this script."
    )

if str(SILENT_FACE_DIR) not in sys.path:
    sys.path.append(str(SILENT_FACE_DIR))

try:
    from src.anti_spoof_predict import AntiSpoofPredict
    from src.generate_patches import CropImage
    from src.utility import parse_model_name
except ImportError as e:
    print(f"Error importing Silent-Face-Anti-Spoofing modules: {e}")
    print(f"Current sys.path: {sys.path}")
    raise

class SpoofDetector:
    def check_gpu_availability(self):
        """Check GPU availability and print detailed information"""
        print("\nGPU Status Check:")
        
        # Check PyTorch CUDA
        torch_cuda = torch.cuda.is_available()
        print(f"PyTorch CUDA Available: {torch_cuda}")
        if torch_cuda:
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        
        # Check OpenCV CUDA
        opencv_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        print(f"OpenCV CUDA Available: {opencv_cuda}")
        if opencv_cuda:
            try:
                # Test OpenCV CUDA
                test_mat = cv2.cuda.GpuMat()
                print("OpenCV CUDA Test: Success")
            except Exception as e:
                print(f"OpenCV CUDA Test Failed: {e}")
        
        # Check CUDA environment
        cuda_path = os.environ.get('CUDA_PATH')
        print(f"CUDA_PATH: {cuda_path}")
        
        return torch_cuda and opencv_cuda
    
    def __init__(self, device_id=0, model_dir=None):
        """Initialize spoof detector with device and model directory"""
        # Check PyTorch GPU availability
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            self.device_id = device_id
            print(f"PyTorch using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            self.device_id = -1
            print("PyTorch using CPU")
        
        # Set default model directory relative to Silent Face directory
        if model_dir is None:
            model_dir = str(SILENT_FACE_DIR / "resources/anti_spoof_models")
        self.model_dir = model_dir
        
        # Set up detection model paths
        detection_model_dir = SILENT_FACE_DIR / "resources/detection_model"
        caffemodel_path = str(detection_model_dir / "Widerface-RetinaFace.caffemodel")
        deploy_path = str(detection_model_dir / "deploy.prototxt")
        
        if not os.path.exists(self.model_dir):
            raise RuntimeError(
                f"Model directory not found at {self.model_dir}! "
                "Please ensure the anti-spoof models are installed correctly."
            )
            
        if not os.path.exists(caffemodel_path) or not os.path.exists(deploy_path):
            raise RuntimeError(
                f"Detection model files not found at {detection_model_dir}! "
                "Please ensure all model files are present."
            )
            
        try:
            self.model_test = AntiSpoofPredict(
                self.device_id,
                deploy_path, 
                caffemodel_path
            )
            self.image_cropper = CropImage()
            
            # Print device being used
            print(f"Neural network running on: {self.model_test.device}")
            
        except Exception as e:
            print(f"Error initializing spoof detector: {e}")
            raise
    
    def predict(self, image):
        try:
            # Create a copy for visualization
            viz_image = image.copy()
            
            # Get all face bboxes
            image_bboxes = self.model_test.get_bbox_batch(image)
            if image_bboxes is None or len(image_bboxes) == 0:
                return True, 0.0, None, image
            
            # Process single model for speed
            model_name = os.listdir(self.model_dir)[0]
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            model_path = os.path.join(self.model_dir, model_name)
            
            # Track results for all faces
            spoof_detected = False
            results = []
            
            # Process each face
            for image_bbox in image_bboxes:
                # Prepare image for current face
                param = {
                    "org_img": image,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True if scale is not None else False
                }
                
                img = self.image_cropper.crop(**param)
                
                # Get prediction using GPU
                with torch.cuda.amp.autocast():
                    prediction = self.model_test.predict(img, model_path)
                
                # Get result for current face
                label = np.argmax(prediction)
                confidence = (prediction[0][label]) * 100
                is_real = label == 1
                
                results.append({
                    'bbox': image_bbox,
                    'is_real': is_real,
                    'confidence': confidence
                })
                
                # Draw boxes and labels for each face
                x, y, w, h = image_bbox
                
                # Ensure box fits within image bounds
                frame_h, frame_w = viz_image.shape[:2]
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame_w - x)
                h = min(h, frame_h - y)
                
                if not is_real:
                    spoof_detected = True
                    # Draw face box in red for fake faces
                    cv2.rectangle(viz_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    
                    # Prepare warning text
                    text = f"FAKE ({confidence:.1f}%)"
                    
                    # Calculate text size and position
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = min(frame_w / 1000, 0.7)
                    thickness = 2
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    
                    # Position text above face if possible, below if not
                    text_y = y - 10 if y > text_size[1] + 10 else y + h + text_size[1] + 10
                    
                    # Draw text background
                    padding = 5
                    cv2.rectangle(viz_image, 
                                (x, text_y - text_size[1] - padding),
                                (x + text_size[0] + padding, text_y + padding),
                                (0, 0, 255), cv2.FILLED)
                    
                    # Draw text
                    cv2.putText(viz_image, text,
                              (x + padding//2, text_y),
                              font, font_scale, (255, 255, 255), thickness)
                else:
                    # Draw thin green box for real faces
                    cv2.rectangle(viz_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
            # Return results
            return True, confidence, results, viz_image
            
        except Exception as e:
            print(f"Error in spoof detection: {str(e)}")
            return True, 0.0, None, image