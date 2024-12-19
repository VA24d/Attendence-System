#face detection works on multiple images
#spoof detection works on multiple images too 
# buttons are working[start recognition, register new face, check face, check image, quit]
#new button to check frame, moved few gui components to new py file, added dark mode

import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
# import onnxruntime as ort
from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import sys
import time
import torch
from datetime import datetime
import hashlib
import shutil
from gui_components import GUI
from theme_utils import ThemeManager
from ui_handlers import UIEventHandlers

# Add Silent-Face-Anti-Spoofing directory to Python path
SILENT_FACE_DIR = Path(__file__).parent / "Silent-Face-Anti-Spoofing-master"
if not SILENT_FACE_DIR.exists():
    raise RuntimeError(
        "Silent-Face-Anti-Spoofing-master directory not found! "
        "Please ensure it's in the same directory as this script."
    )
sys.path.append(str(SILENT_FACE_DIR))
# Now import spoof detector
from spoof_detector import SpoofDetector

# At the top of the file, after imports
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['OPENCV_DNN_BACKEND'] = 'CUDA'
os.environ['OPENCV_DNN_TARGET'] = 'CUDA'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Suppress the specific FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

# Device configuration - read from config file if exists
CONFIG_FILE = Path(__file__).parent / "config.txt"
try:
    with open(CONFIG_FILE, 'r') as f:
        DEVICE = f.read().strip().lower()
except FileNotFoundError:
    DEVICE = "cpu"  # Default to cpu if no config file

# If you want to force CPU usage, change this line
if not torch.cuda.is_available():
    DEVICE = "cpu"  # Fall back to CPU if CUDA is not available

class AttendanceSystem:
    def __init__(self):
        self.root = tk.Tk()
        # self.root.configure(bg="lightblue")
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1024x768")
        
        # Add theme state
        self.dark_mode = tk.BooleanVar(value=False)
        
        # Initialize theme and UI handlers
        # self.colors = ThemeManager.get_color_schemes()
        self.ui_handlers = UIEventHandlers(self)
        
        # Initialize device variable for radio buttons with current device
        self.device_var = tk.StringVar(value=DEVICE.upper())
        
        # Check device availability and setup
        self.setup_device()
        
        # Initialize face analyzer with selected device
        self.setup_face_analyzer()
        
        # Initialize face embeddings
        self.known_face_embeddings = []
        self.known_face_ids = []
        
        # Initialize GUI
        self.gui = GUI(self)
        
        # Update student list immediately after GUI initialization
        self.ui_handlers.update_student_list()
        
        # Then load face encodings
        self.load_face_embeddings()
        
        # Initialize camera as None
        self.cap = None
        self.is_recognition_active = False
        
        # Initialize camera with error handling
        try:
            self.cap = self.init_camera()
        except Exception as e:
            print(f"Warning: Failed to initialize camera: {e}")
        
        # Initialize spoof detector with same device as face analyzer
        device_id = -1  # Default to CPU
        if self.device == "gpu" and torch.cuda.is_available():
            device_id = 0
        self.spoof_detector = SpoofDetector(device_id=device_id)
        
        # Add spoof detection status
        self.last_spoof_check = 0
        self.spoof_check_interval = 1.0  # Check every 1 second
        
        # Initialize video-related variables
        self.current_frame = None
        self.current_face_data = None
        
        self.set_app_icon()

        # Bind window close button (X) to quit function
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)
        
        # Bind escape key to quit function
        self.root.bind('<Escape>', lambda e: self.quit_application())
        
        self.last_resize_time = 0
        self.resize_debounce_delay = 0.1  # 100ms
        self.preview_dimensions = (640, 480)  # Default size
        
        # Bind resize event
        self.root.bind("<Configure>", self.handle_window_resize)
        
        self.last_detections = {}  # Track multiple people
        self.detection_cleanup_interval = 10  # Cleanup every 10 seconds
        self.last_cleanup = time.time()
        
        # force focus on the window
        self.root.focus_force()
        
    def set_app_icon(self):
        # Load the icon (should be a .png file)
        icon = Image.open("logo.png")
        photo = ImageTk.PhotoImage(icon)

        # Set the icon for the Tkinter app
        self.root.iconphoto(True, photo)
    
    def setup_device(self):
        """Setup and verify device configuration"""
        self.device = DEVICE.lower()
        
        if self.device == "gpu":
            try:
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    print("\nGPU Detection:")
                    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
                    print(f"CUDA Version: {torch.version.cuda}")
                    
                    try:
                        # Configure OpenCV DNN for CUDA
                        cv2.cuda.setDevice(0)
                        cv2.ocl.setUseOpenCL(False)
                        cv2.setNumThreads(1)
                        print("OpenCV CUDA backend configured")
                    except Exception as e:
                        print(f"OpenCV CUDA setup failed: {e}")
                    
                    self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                else:
                    print("\nGPU requested but PyTorch CUDA not available")
                    print("Falling back to CPU")
                    self.device = "cpu"
                    self.providers = ['CPUExecutionProvider']
            except Exception as e:
                print(f"\nError checking GPU: {str(e)}")
                print("Falling back to CPU")
                self.device = "cpu"
                self.providers = ['CPUExecutionProvider']
        if self.device == "mps":
            # Apple metal plugin
            self.providers = ['MetalExecutionProvider']
        else:
            print("\nUsing CPU for processing (GPU not requested)")
            self.providers = ['CPUExecutionProvider']
        
        print(f"Device set to: {self.device.upper()}")
    
    def setup_face_analyzer(self):
        """Initialize InsightFace analyzer with selected device"""
        self.face_analyzer = FaceAnalysis(
            name='buffalo_l',
            providers=self.providers
        )
        self.face_analyzer.prepare(ctx_id=0 if self.device == "gpu" else -1, 
                                 det_size=(640, 640))
        print(f"Face analyzer initialized with {self.device.upper()}")
    
    def load_face_embeddings(self):
        """Load face embeddings from student_images directory"""
        base_path = Path("assets/student_images")
        if not base_path.exists():
            messagebox.showerror("Error", "student_images directory not found!")
            return
        
        print("\nLoading face embeddings...")
        
        # Clear existing embeddings
        self.known_face_embeddings = []
        self.known_face_ids = []
        
        # Load each student's images
        for student_dir in base_path.iterdir():
            if student_dir.is_dir():
                student_id = student_dir.name
                # Process each image in student's directory
                for img_path in student_dir.glob("*.jpg"):
                    try:
                        # Load and process image
                        img = cv2.imread(str(img_path))
                        if img is None:
                            print(f"Could not read {img_path}")
                            continue
                        
                        # Get face embedding
                        faces = self.face_analyzer.get(img)
                        
                        if len(faces) > 0:
                            # Get the face with highest detection score
                            face = max(faces, key=lambda x: x.det_score)
                            embedding = face.embedding
                            
                            self.known_face_embeddings.append(embedding)
                            self.known_face_ids.append(student_id)
                            print(f"Loaded embedding for student {student_id} from {img_path.name}")
                        else:
                            print(f"No face found in {img_path}")
                            
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
        
        # Convert embeddings list to numpy array for faster processing
        self.known_face_embeddings = np.array(self.known_face_embeddings)
        print(f"\nFinished loading {len(self.known_face_ids)} face embeddings")
        
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Loaded {len(self.known_face_ids)} face embeddings")
    
    def init_camera(self):
        """Initialize camera with error handling"""
        # First ensure any existing camera is released
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
                
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not access webcam. Please check if it's connected properly.")
                return None
            
            # Test read a frame
            ret, _ = cap.read()
            if not ret:
                cap.release()
                messagebox.showerror("Error", "Could not read from webcam.")
                return None
            
            return cap
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            messagebox.showerror("Error", f"Camera initialization failed: {str(e)}")
            return None
    
    def run(self):
        self.root.mainloop()
    
    def __del__(self):
        """Improved destructor with proper error handling"""
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                try:
                    self.cap.release()
                except Exception as e:
                    print(f"Warning: Error releasing camera in destructor: {e}")
        except Exception as e:
            print(f"Warning: Error in destructor: {e}")
    
    def train_mode(self):
        """Switch to training mode"""
        self.load_and_train_embeddings()
        messagebox.showinfo("Training Complete", "Face embeddings have been updated and saved!")
    
    def recognition_mode(self):
        """Switch to recognition mode"""
        self.load_stored_embeddings()
        messagebox.showinfo("Recognition Mode", "Loaded stored face embeddings!")
    
    def load_and_train_embeddings(self):
        """Train and save face embeddings from student images with batch processing"""
        if self.device == "cpu":
            response = messagebox.askquestion(
                "CPU Training",
                "Training on CPU can be significantly slower. Continue?",
                icon='warning'
            )
            if response != 'yes':
                return
        
        base_path = Path("assets/student_images")
        embeddings_path = Path("assets/face_encodings")
        
        if not base_path.exists():
            messagebox.showerror("Error", "student_images directory not found!")
            return
        
        embeddings_path.mkdir(exist_ok=True)
        print("\nTraining mode: Processing face embeddings...")
        
        self.known_face_embeddings = []
        self.known_face_ids = []
        
        BATCH_SIZE = 8 if self.device == "gpu" else 4
        
        for student_dir in base_path.iterdir():
            if student_dir.is_dir():
                student_id = student_dir.name
                student_images = []
                image_paths = list(student_dir.glob("*.jpg"))
                
                # Load all images first
                for img_path in image_paths:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            student_images.append(img)
                        else:
                            print(f"Could not read {img_path}")
                    except Exception as e:
                        print(f"Error loading {img_path}: {str(e)}")
                
                if not student_images:
                    continue
                
                # Process images in batches
                student_embeddings = []
                for i in range(0, len(student_images), BATCH_SIZE):
                    batch = student_images[i:min(i + BATCH_SIZE, len(student_images))]
                    try:
                        # Process batch
                        faces_batch = [self.face_analyzer.get(img) for img in batch]
                        
                        # Extract embeddings from valid faces
                        for faces in faces_batch:
                            if len(faces) > 0:
                                # Get the face with highest detection score
                                face = max(faces, key=lambda x: x.det_score)
                                student_embeddings.append(face.embedding)
                    
                    except Exception as e:
                        print(f"Error processing batch: {str(e)}")
                
                if student_embeddings:
                    # Save embeddings for this student
                    student_embeddings = np.array(student_embeddings)
                    np.save(embeddings_path / f"{student_id}_embeddings.npy", student_embeddings)
                    
                    # Add to current session
                    self.known_face_embeddings.extend(student_embeddings)
                    self.known_face_ids.extend([student_id] * len(student_embeddings))
                    print(f"Processed {len(student_embeddings)} images for student {student_id}")
        
        # Convert to numpy array for faster processing
        self.known_face_embeddings = np.array(self.known_face_embeddings)
        print(f"\nTraining complete! Saved embeddings for {len(set(self.known_face_ids))} students")
        
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Trained {len(self.known_face_ids)} face embeddings")
    
    def load_stored_embeddings(self):
        """Load pre-computed face embeddings with memory optimization"""
        embeddings_path = Path("assets/face_encodings")
        
        if not embeddings_path.exists():
            messagebox.showerror("Error", "No stored embeddings found! Please run training first.")
            return
        
        print("\nLoading stored face embeddings...")
        
        # Clear existing embeddings
        self.known_face_embeddings = []
        self.known_face_ids = []
        
        total_size = 0
        # First pass: calculate total size needed
        for embedding_file in embeddings_path.glob("*_embeddings.npy"):
            total_size += os.path.getsize(embedding_file)
        
        # Check if we need memory mapping
        use_mmap = total_size > 1024 * 1024 * 1024  # Use mmap if total size > 1GB
        
        # Load embeddings
        for embedding_file in embeddings_path.glob("*_embeddings.npy"):
            try:
                student_id = embedding_file.stem.replace("_embeddings", "")
                if use_mmap:
                    # Memory-mapped reading for large files
                    embeddings = np.load(embedding_file, mmap_mode='r')
                else:
                    # Direct loading for smaller files
                    embeddings = np.load(embedding_file)
                
                self.known_face_embeddings.extend(embeddings)
                self.known_face_ids.extend([student_id] * len(embeddings))
                print(f"Loaded {len(embeddings)} embeddings for student {student_id}")
                
            except Exception as e:
                print(f"Error loading {embedding_file}: {str(e)}")
        
        # Convert to numpy array for faster processing
        self.known_face_embeddings = np.array(self.known_face_embeddings)
        print(f"\nLoaded {len(self.known_face_ids)} stored face embeddings")
        
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Loaded {len(self.known_face_ids)} face embeddings")
    
    def change_device(self):
        """Handle device change"""
        new_device = self.device_var.get().lower()
        if new_device != self.device:
            response = messagebox.askquestion(
                "Change Device",
                f"Changing to {new_device.upper()} requires restart. Continue?",
                icon='warning'
            )
            if response == 'yes':
                try:
                    # Save new device selection to config file
                    with open(CONFIG_FILE, 'w') as f:
                        f.write(new_device)
                    
                    # Restart application
                    self.root.quit()
                    python = sys.executable
                    os.execl(python, python, *sys.argv)
                except Exception as e:
                    messagebox.showerror(
                        "Error", 
                        f"Failed to change device: {str(e)}"
                    )
            else:
                # Revert selection in GUI
                self.device_var.set(self.device.upper())
    
    def start_registration(self):
        """Start the registration process for a new face"""
        # First ensure any existing camera session is closed
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Get user name/ID
        name = self.get_user_name()
        if not name:
            return
        
        # Initialize camera for registration
        self.cap = self.init_camera()
        if self.cap is None:
            messagebox.showerror("Error", "Could not access camera!")
            return
        
        # Start camera for registration without creating directory yet
        self.register_photos(name)
    
    def get_user_name(self):
        """Show dialog to get user name/ID"""
        dialog = tk.Toplevel(self.root)
        dialog.title("New User Registration")
        dialog.geometry("300x150")
        
        tk.Label(dialog, text="Enter your name/ID:").pack(pady=10)
        
        entry = tk.Entry(dialog, width=30)
        entry.pack(pady=10)
        
        result = [None]  # Use list to store result
        
        def submit():
            name = entry.get().strip()
            if name:
                result[0] = name
                dialog.destroy()
        
        tk.Button(
            dialog,
            text="Submit",
            command=submit,
        ).pack(pady=10)
        
        # Make dialog modal
        dialog.transient(self.root)
        dialog.grab_set()
        self.root.wait_window(dialog)
        
        return result[0]

    def register_photos(self, name):
        """Capture and save photos for registration"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = self.init_camera()
        
        if self.cap is None:
            messagebox.showerror("Error", "Camera not available!")
            return
        
        # Store captured images temporarily
        captured_frames = []
        
        # Create registration window with fixed size
        reg_window = tk.Toplevel(self.root)
        reg_window.title("Face Registration")
        reg_window.geometry("640x720")
        reg_window.resizable(False, False)
        
        def on_reg_window_close():
            """Handle registration window closing"""
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            reg_window.destroy()
            
            # Clean up captured frames
            for frame in captured_frames:
                del frame
                
            # release spacebar and escape binding
            reg_window.unbind('<space>')
            reg_window.unbind('<Escape>')
            
            # bind escape key to close main window
            self.root.bind('<Escape>', lambda e: self.quit_application())
        
        # Bind window close event
        reg_window.protocol("WM_DELETE_WINDOW", on_reg_window_close)
        
        # Main container with padding
        main_container = tk.Frame(reg_window, padx=10, pady=5)
        main_container.pack(fill="both", expand=True)
        
        # Instructions Frame at top
        instruction_frame = tk.LabelFrame(main_container, text="Instructions", padx=10, pady=5)
        instruction_frame.pack(fill="x", pady=5)
        
        instructions = [
            "1. Look straight at the camera",
            "2. Turn your head slightly to the left",
            "3. Turn your head slightly to the right"
        ]
        current_pose = 0
        
        # Main instruction
        instruction_label = tk.Label(
            instruction_frame,
            text=instructions[current_pose],
            font=('Arial', 12, 'bold')
        )
        instruction_label.pack(pady=2)
        
        # Additional instruction
        tk.Label(
            instruction_frame,
            text="Position your face in the green box and click 'Take Photo'",
            font=('Arial', 10)
        ).pack(pady=2)
        
        # Video display - fixed size
        video_frame = tk.Frame(main_container)
        video_frame.pack(pady=5)
        
        video_label = tk.Label(video_frame, width=600, height=400)  # Fixed size
        video_label.pack()
        
        # Progress Frame
        progress_frame = tk.Frame(main_container)
        progress_frame.pack(fill="x", pady=5)
        
        # Progress indicators
        progress_labels = []
        for i in range(3):
            label = tk.Label(
                progress_frame,
                text=f"Photo {i+1}",
                fg='gray',
                font=('Arial', 10)
            )
            label.pack(side=tk.LEFT, expand=True)
            progress_labels.append(label)
        
        # Status label
        status_label = tk.Label(
            main_container,
            text="Ready to capture photo 1 of 3",
            font=('Arial', 11)
        )
        status_label.pack(pady=5)
        
        # Buttons Frame at bottom
        button_frame = tk.Frame(main_container)
        button_frame.pack(pady=5)
        
        def update_preview():
            if len(captured_frames) >= 3:
                return
            
            ret, frame = self.cap.read()
            if ret:
                # Resize frame to fit display
                frame = cv2.resize(frame, (600, 400))
                
                # Draw face detection box
                faces = self.face_analyzer.get(frame)
                for face in faces:
                    bbox = face.bbox.astype(int)
                    # Use red box for multiple faces, green for single face
                    color = (0, 255, 0) if len(faces) == 1 else (0, 0, 255)
                    cv2.rectangle(frame, 
                                (bbox[0], bbox[1]), 
                                (bbox[2], bbox[3]), 
                                color, 2)
                
                # Convert frame for display
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                img = ImageTk.PhotoImage(img)
                video_label.imgtk = img
                video_label.configure(image=img)
            
            if len(captured_frames) < 3:  # Continue preview until all photos are captured
                reg_window.after(10, update_preview)
        
        def capture_photo():
            nonlocal current_pose
            if current_pose >= 3:
                return
            
            ret, frame = self.cap.read()
            if ret:
                faces = self.face_analyzer.get(frame)
                if len(faces) == 0:
                    messagebox.showwarning("Warning", "No face detected! Please try again.")
                    return
                if len(faces) > 1:
                    messagebox.showwarning(
                        "Warning", 
                        "Multiple faces detected! Please ensure only one person is in frame."
                    )
                    return
                
                # Store frame in memory
                captured_frames.append(frame.copy())
                
                # Update progress indicator
                progress_labels[current_pose].config(fg='green', font=('Arial', 10, 'bold'))
                
                current_pose += 1
                if current_pose < 3:
                    instruction_label.config(text=instructions[current_pose])
                    status_label.config(text=f"Ready to capture photo {current_pose + 1} of 3")
                    capture_btn.config(text=f"Take Photo {current_pose + 1}")
                else:
                    finish_registration()
        
        def finish_registration():
            if len(captured_frames) < 3:
                messagebox.showerror("Error", "All three photos must be captured!")
                return
            
            try:
                # Create directory for new user only after all photos are captured
                user_dir = Path("assets/student_images") / name
                if user_dir.exists():
                    messagebox.showerror("Error", f"User {name} already exists!")
                    return
                
                user_dir.mkdir(parents=True, exist_ok=True)
                
                # Save all captured frames
                for i, frame in enumerate(captured_frames):
                    img_path = user_dir / f"pose_{i + 1}.jpg"
                    cv2.imwrite(str(img_path), frame)
                
                status_label.config(text="Registration complete!")
                capture_btn.config(state='disabled')
                
                response = messagebox.askyesno(
                    "Registration Complete",
                    "Photos captured successfully! Would you like to train the system now?"
                )
                
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                
                reg_window.destroy()
                
                if response:
                    self.train_mode()
                    self.update_student_list()
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save images: {str(e)}")
                if user_dir.exists():
                    try:
                        shutil.rmtree(user_dir)  # Clean up on failure
                    except:
                        pass
                    
        # bind space key to capture photo
        reg_window.bind('<space>', lambda e: capture_photo())
        
        # Capture button
        capture_btn = tk.Button(
            button_frame,
            text="Take Photo 1",
            command=capture_photo,
            bg='lightgray',
            fg='black',
            font=('Arial', 11, 'bold'),
            width=15,
            height=1
        )
        capture_btn.pack(side=tk.LEFT, padx=5)
        
        # bind escape key to close registration window
        reg_window.bind('<Escape>', lambda e: on_reg_window_close())
        
        # Cancel button
        tk.Button(
            button_frame,
            text="Cancel",
            command=reg_window.destroy,
            bg='#f44336',
            fg='black',
            font=('Arial', 11),
            width=8,
            height=1
        ).pack(side=tk.LEFT, padx=5)
        
        # Start preview
        update_preview()
    
    def check_image(self):
        """Upload and check an image against the database"""
        from tkinter import filedialog
        
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Read and process image
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Error", "Could not read image file!")
                return
            
            # Check for spoofing first
            is_real, spoof_confidence, bbox, viz_img = self.spoof_detector.predict(img)
            if not is_real:
                # Show the visualization in a warning window
                result_window = tk.Toplevel(self.root)
                result_window.title("Spoof Detection Result")
                
                # Resize image for display
                height, width = viz_img.shape[:2]
                max_size = 800
                if width > height:
                    new_width = min(width, max_size)
                    new_height = int(height * (new_width / width))
                else:
                    new_height = min(height, max_size)
                    new_width = int(width * (new_height / height))
                
                viz_img = cv2.resize(viz_img, (new_width, new_height))
                cv2image = cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                img = ImageTk.PhotoImage(image=img)
                
                # Display result
                label = tk.Label(result_window, image=img)
                label.image = img
                label.pack(pady=10)
                
                # Add warning text
                tk.Label(
                    result_window,
                    text=f"This appears to be a fake/spoofed image!\nConfidence: {spoof_confidence:.1f}%",
                    font=('Arial', 12, 'bold'),
                    fg='red'
                ).pack(pady=5)
                
                # Close button
                tk.Button(
                    result_window,
                    text="Close",
                    command=result_window.destroy,
                    bg='lightgray',
                    fg='black',
                    font=('Arial', 11)
                ).pack(pady=5)
                
                return
            
            # Calculate aspect ratio preserving resize
            max_width = 800
            max_height = 600
            height, width = img.shape[:2]
            
            # Calculate new size maintaining aspect ratio
            if width > height:
                new_width = min(width, max_width)
                new_height = int(height * (new_width / width))
                if new_height > max_height:
                    new_height = max_height
                    new_width = int(width * (new_height / height))
            else:
                new_height = min(height, max_height)
                new_width = int(width * (new_height / height))
                if new_width > max_width:
                    new_width = max_width
                    new_height = int(height * (new_width / width))
            
            # Create result window with dynamic size
            result_window = tk.Toplevel(self.root)
            result_window.title("Recognition Result")
            result_window.geometry(f"{new_width + 40}x{new_height + 100}")  # Add padding for button
            
            # Process image
            faces = self.face_analyzer.get(img)
            if len(faces) == 0:
                messagebox.showwarning("Warning", "No faces detected in image!")
                result_window.destroy()
                return
            
            # Draw boxes and find matches
            for face in faces:
                bbox = face.bbox.astype(int)
                embedding = face.embedding
                
                if len(self.known_face_embeddings) > 0:
                    # Find best match
                    curr_embedding = embedding.reshape(1, -1)
                    similarities = cosine_similarity(curr_embedding, self.known_face_embeddings)[0]
                    best_match_index = similarities.argmax()
                    max_similarity = similarities[best_match_index]
                    
                    # Get top 3 matches for verification
                    top_indices = np.argsort(similarities)[-3:][::-1]
                    top_ids = [self.known_face_ids[i] for i in top_indices]
                    
                    # Check if matches are consistent
                    is_consistent = (len(set(top_ids)) == 1)
                    confidence = round(max_similarity * 100, 1)
                    
                    if max_similarity > 0.5 and is_consistent:
                        # Match found
                        student_id = self.known_face_ids[best_match_index]
                        color = (0, 255, 0)  # Green
                        text = f"{student_id} ({confidence}%)"
                    else:
                        # No match
                        color = (0, 0, 255)  # Red
                        text = "Unknown"
                    
                    # Draw rectangle
                    cv2.rectangle(img, 
                                (bbox[0], bbox[1]), 
                                (bbox[2], bbox[3]), 
                                color, 3)
                    
                    # Enhanced text display with larger font
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 1.5  # Increased from 1.0 to 1.5
                    thickness = 3    # Increased from 2 to 3
                    
                    # Calculate text size for background
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    
                    # Add more padding around text
                    padding_x = 15  # Horizontal padding
                    padding_y = 15  # Vertical padding
                    
                    # Draw text background with more padding
                    cv2.rectangle(img, 
                                (bbox[0] - 5, bbox[1] - text_size[1] - padding_y),
                                (bbox[0] + text_size[0] + padding_x, bbox[1]),
                                color, cv2.FILLED)
                    
                    # Draw text
                    cv2.putText(img, text,
                               (bbox[0] + 5, bbox[1] - padding_y//2),  # Adjusted y position
                               font, font_scale,
                               (255, 255, 255), thickness)
            
            # Resize image maintaining aspect ratio
            img = cv2.resize(img, (new_width, new_height))
            cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            img = ImageTk.PhotoImage(img)
            
            # Display result
            label = tk.Label(result_window, image=img)
            label.image = img  # Keep a reference
            label.pack(pady=10)
            
            # Close button
            tk.Button(
                result_window,
                text="Close",
                command=result_window.destroy,
                bg='#f44336',
                fg='black',
                font=('Arial', 11)
            ).pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
    
    def display_frame(self, frame):
        """Helper method to display a frame"""
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def quit_application(self):
        """Gracefully shutdown the application"""
        try:
            # Stop recognition if active
            if self.is_recognition_active:
                self.toggle_recognition()
            
            # Release camera
            if self.cap is not None:
                try:
                    self.cap.release()
                    self.cap = None
                except Exception as e:
                    print(f"Warning: Error releasing camera: {e}")
            
            # Release CUDA memory if using GPU
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Warning: Error clearing CUDA memory: {e}")
            
            # Destroy the window
            try:
                self.root.quit()
                self.root.destroy()
                print("Application closed successfully")
            except Exception as e:
                print(f"Warning: Error destroying window: {e}")
                sys.exit(0)  # Force exit if needed
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
            try:
                self.root.quit()
                sys.exit(0)
            except Exception as e:
                print(f"Fatal error during shutdown: {e}")
    
    def add_to_database(self):
        """Add current frame to database for recognized face"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            messagebox.showerror("Error", "No frame available!")
            return
        
        # Check for multiple faces in the frame
        faces = self.face_analyzer.get(self.current_frame)
        if len(faces) == 0:
            messagebox.showwarning("Warning", "No face detected in current frame!")
            return
        if len(faces) > 1:
            messagebox.showwarning("Warning", "Multiple faces detected! Please ensure only one person is in frame.")
            return
        
        if not hasattr(self, 'current_face_data') or not self.current_face_data:
            messagebox.showwarning("Warning", "No face detected in current frame!")
            return
        
        face_data = self.current_face_data
        if face_data.get('is_unknown', True):
            messagebox.showwarning(
                "Unknown Face", 
                "Face not recognized! Please use 'Register New Face' for new individuals."
            )
            return
        
        student_id = face_data['student_id']
        self.save_frame_to_student(student_id, self.current_frame.copy())
        messagebox.showinfo("Success", f"Frame added to {student_id}'s dataset!")
    
    def handle_mismatch(self):
        """Handle mismatched face detection"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            messagebox.showerror("Error", "No frame available!")
            return
        
        # Check for multiple faces in the frame
        faces = self.face_analyzer.get(self.current_frame)
        if len(faces) == 0:
            messagebox.showwarning("Warning", "No face detected in current frame!")
            return
        if len(faces) > 1:
            messagebox.showwarning("Warning", "Multiple faces detected! Please ensure only one person is in frame.")
            return
        
        if not hasattr(self, 'current_face_data') or not self.current_face_data:
            messagebox.showwarning("Warning", "No face detected in current frame!")
            return
        
        face_data = self.current_face_data
        if face_data.get('is_unknown', True):
            messagebox.showwarning(
                "Unknown Face", 
                "Face not recognized! Please use 'Register New Face' for new individuals."
            )
            return
        
        # Create student selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Correct Student")
        dialog.geometry("300x400")
        
        # Add instructions
        tk.Label(
            dialog,
            text="Select the correct student for this face:",
            wraplength=250
        ).pack(pady=10)
        
        # Create listbox with scrollbar
        frame = tk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(
            frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=listbox.yview)
        
        # Populate student list
        student_ids = sorted(set(self.known_face_ids))
        for student_id in student_ids:
            listbox.insert(tk.END, student_id)
        
        def on_select():
            if listbox.curselection():
                selected_id = listbox.get(listbox.curselection())
                self.save_frame_to_student(selected_id, self.current_frame.copy())
                messagebox.showinfo("Success", f"Frame added to {selected_id}'s dataset!")
                dialog.destroy()
        
        # Add buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        tk.Button(
            button_frame,
            text="Confirm",
            command=on_select,
            bg='white',
            fg='black'
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
            bg='white',
            fg='black'
        ).pack(side=tk.LEFT, padx=5)
    
    def save_frame_to_student(self, student_id, frame):
        """Save frame to student's directory with timestamp"""
        # Create student directory if it doesn't exist
        student_dir = Path("assets/student_images") / student_id
        student_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate frame hash to prevent duplicates
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        frame_hash = hashlib.md5(frame_bytes).hexdigest()[:8]
        
        # Save frame
        filename = f"{timestamp}_{frame_hash}.jpg"
        filepath = student_dir / filename
        
        # Save with compression
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    def draw_unknown_face_box(self, frame, bbox):
        """Draw box around unknown face"""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        text = "Unknown Face"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        thickness = 1
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Draw label background
        cv2.rectangle(frame, 
                     (x1, y2 - text_size[1] - 10),
                     (x1 + text_size[0] + 10, y2),
                     (0, 0, 255), cv2.FILLED)
        
        # Add text
        cv2.putText(frame, text,
                    (x1 + 5, y2 - 5),
                    font, font_scale, (255, 255, 255), thickness)
    
    def toggle_theme(self, event=None):
        """Handle theme toggle"""
        self.dark_mode.set(not self.dark_mode.get())
        self.gui.draw_toggle_switch()
        # self.gui.update_colors()

    def update_student_list(self):
        """Update the list of available students"""
        base_path = Path("assets/student_images")
        if base_path.exists():
            # Get list of student IDs (folder names)
            student_ids = [folder.name for folder in base_path.iterdir() if folder.is_dir()]
            student_ids.sort()  # Sort IDs
            
            # Update combobox values
            self.gui.student_selector.configure(values=['All'] + student_ids)
            
            # Set to 'All' by default
            self.gui.student_selector.set('All')
            
            print(f"Found student IDs: {', '.join(student_ids)}")
        else:
            print("Student images directory not found")

    def toggle_recognition(self):
        """Modified to ensure clean shutdown of recognition"""
        try:
            self.is_recognition_active = not self.is_recognition_active
            if self.is_recognition_active:
                if self.cap is None or not self.cap.isOpened():
                    self.cap = self.init_camera()
                    if self.cap is None:
                        return
                
                # Do initial resize before starting video feed
                self.root.update_idletasks()  # Ensure window dimensions are updated
                padding = 40
                margin = 20
                available_width = self.root.winfo_width() - padding
                available_height = self.root.winfo_height() - self.gui.video_label.winfo_y() - margin
                
                self.resize_video_frame(available_width, available_height)
            
                self.gui.toggle_btn.config(text="Stop Recognition", bg='#f44336')
                # Show database operation buttons
                self.gui.add_db_btn.pack(side=tk.LEFT, padx=5)
                self.gui.mismatch_btn.pack(side=tk.LEFT, padx=5)
                
                self.update_video_feed()
            else:
                self.gui.toggle_btn.config(text="Start Recognition", bg='#2196F3')
                # Hide database operation buttons
                self.gui.add_db_btn.pack_forget()
                self.gui.mismatch_btn.pack_forget()
                # Ensure camera is released when stopping
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
        except Exception as e:
            print(f"Error toggling recognition: {e}")
            self.is_recognition_active = False
            self.gui.toggle_btn.config(text="Start Recognition", bg='#2196F3')
            # Hide database operation buttons
            self.gui.add_db_btn.pack_forget()
            self.gui.mismatch_btn.pack_forget()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
    
    def resize_video_frame(self, width, height):
        """Resize video frame while maintaining aspect ratio with bounds checking"""
        if not hasattr(self, 'cap') or self.cap is None:
            return
            
        # Minimum and maximum dimensions
        MIN_WIDTH = 320
        MIN_HEIGHT = 240
        MAX_WIDTH = 1920
        MAX_HEIGHT = 1080
        
        # Constrain input dimensions
        width = max(MIN_WIDTH, min(width, MAX_WIDTH))
        height = max(MIN_HEIGHT, min(height, MAX_HEIGHT))
        
        try:
            # Get camera aspect ratio
            cam_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            cam_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if cam_width <= 0 or cam_height <= 0:
                raise ValueError("Invalid camera dimensions")
                
            cam_aspect = cam_width / cam_height
            
            # Calculate new dimensions maintaining aspect ratio
            if width / height > cam_aspect:
                new_height = height
                new_width = int(height * cam_aspect)
            else:
                new_width = width
                new_height = int(width / cam_aspect)
            
            # Ensure dimensions are even (required by some codecs)
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            # Store new dimensions
            self.video_dimensions = (new_width, new_height)
            
            # Update status if significant change
            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"Video resized to {new_width}x{new_height}")
                
        except Exception as e:
            print(f"Error resizing video frame: {str(e)}")
            # Use safe default dimensions
            self.video_dimensions = (640, 480)

    def handle_window_resize(self, event):
        """Handle window resize with debouncing"""
        # Only handle root window resizes
        if event.widget != self.root:
            return
            
        current_time = time.time()
        if current_time - self.last_resize_time < self.resize_debounce_delay:
            return
            
        self.last_resize_time = current_time
        
        # Calculate available space
        padding = 40  # Total horizontal padding
        margin = 20   # Bottom margin
        
        available_width = self.root.winfo_width() - padding
        available_height = self.root.winfo_height() - self.gui.video_label.winfo_y() - margin
        
        self.update_preview_dimensions(available_width, available_height)

    def update_preview_dimensions(self, width, height):
        """Update preview dimensions while maintaining aspect ratio"""
        # Minimum dimensions
        MIN_WIDTH = 320
        MIN_HEIGHT = 240
        
        # Maximum dimensions
        MAX_WIDTH = 1920
        MAX_HEIGHT = 1080
        
        try:
            if self.cap is None:
                return
                
            # Get camera aspect ratio
            cam_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            cam_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if cam_width <= 0 or cam_height <= 0:
                raise ValueError("Invalid camera dimensions")
                
            # Calculate aspect ratio
            cam_aspect = cam_width / cam_height
            
            # Constrain dimensions
            width = max(MIN_WIDTH, min(width, MAX_WIDTH))
            height = max(MIN_HEIGHT, min(height, MAX_HEIGHT))
            
            # Calculate new dimensions maintaining aspect ratio
            if width / height > cam_aspect:
                new_height = height
                new_width = int(height * cam_aspect)
            else:
                new_width = width
                new_height = int(width / cam_aspect)
            
            # Ensure even dimensions
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            self.preview_dimensions = (new_width, new_height)
            
        except Exception as e:
            print(f"Error updating preview dimensions: {e}")
            self.preview_dimensions = (640, 480)  # Fallback size

    def update_video_feed(self):
        """Update video feed with face recognition and spoof detection"""
        if not self.is_recognition_active or self.cap is None:
            return
        
        try:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to get frame from webcam")
            
            frame = cv2.resize(frame, self.preview_dimensions)
            
            # Store current frame for database operations
            self.current_frame = frame.copy()
            self.current_face_data = None
             
            # Run spoof detection and face recognition in parallel if possible
            if self.device == 'gpu':
                with torch.cuda.stream(torch.cuda.Stream()):
                    is_real, spoof_confidence, face_results, frame = self.spoof_detector.predict(frame)
            else:
                is_real, spoof_confidence, face_results, frame = self.spoof_detector.predict(frame)
            
            # Process face recognition for all faces
            if face_results is not None:
                faces = self.face_analyzer.get(frame)
                selected_id = self.gui.student_selector.get()
                
                # Process all faces regardless of spoof status
                for face, result in zip(faces, face_results):
                    bbox = result['bbox']
                    is_fake = not result['is_real']
                    
                    # Process face recognition
                    self.process_face(face, frame, selected_id, is_fake=is_fake)
            
            self.display_frame(frame)
            
        except Exception as e:
            print(f"Error in video processing: {str(e)}")
        
        if self.is_recognition_active:
            self.root.after(1, self.update_video_feed)

    def display_frame(self, frame):
        """Display frame in GUI"""
        """Display resized frame in GUI"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.gui.video_label.imgtk = imgtk
            self.gui.video_label.configure(image=imgtk)
            
        except Exception as e:
            print(f"Error displaying frame: {e}")

    def process_face(self, face, frame, selected_id, is_fake=False):
        """Process face recognition with fake detection support"""
        try:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            
             # Cleanup old detections periodically
            current_time = time.time()
            if current_time - self.last_cleanup > self.detection_cleanup_interval:
                self._cleanup_old_detections()
                self.last_cleanup = current_time
            
            if len(self.known_face_embeddings) > 0:
                # Use GPU for similarity calculations if available
                if self.device == 'gpu':
                    curr_embedding = torch.tensor(embedding.reshape(1, -1)).cuda()
                    known_embeddings = torch.tensor(self.known_face_embeddings).cuda()
                    similarities = torch.nn.functional.cosine_similarity(curr_embedding, known_embeddings)
                    similarities = similarities.cpu().numpy()
                else:
                    curr_embedding = embedding.reshape(1, -1)
                    similarities = cosine_similarity(curr_embedding, self.known_face_embeddings)[0]
                
                best_match_index = similarities.argmax()
                max_similarity = similarities[best_match_index]
                best_match_id = self.known_face_ids[best_match_index]
                
                # Get top matches
                top_indices = np.argsort(similarities)[-3:][::-1]
                top_ids = [self.known_face_ids[i] for i in top_indices]
                
                should_show = (selected_id == 'All' or selected_id == best_match_id)
                is_consistent = (len(set(top_ids)) == 1)
                
                confidence = self.calculate_confidence(max_similarity)
                
                if max_similarity > 0.5 and should_show and is_consistent:
                    # Store face data for database operations
                    self.current_face_data = {
                        'is_unknown': False,
                        'student_id': best_match_id,
                        'confidence': confidence,
                        'bbox': bbox,
                        'is_fake': is_fake
                    }
                    
                    should_print = True
                    if best_match_id in self.last_detections:
                        last_detect = self.last_detections[best_match_id]
                        if (is_fake == last_detect['status'] and 
                            current_time - last_detect['timestamp'] < 5):
                            should_print = False
                    
                    if should_print:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        status = "FAKE" if is_fake else "REAL"
                        print(f"[{timestamp}] Detected: {best_match_id} ({confidence:.1f}%) - {status}")
                        
                        # Update tracking
                        self.last_detections[best_match_id] = {
                            'status': is_fake,
                            'timestamp': current_time
                        }
                    
                    # Draw face detection box and ID
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    
                    # Draw ID text
                    text = f"ID: {best_match_id} ({confidence:.1f}%)"
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    thickness = 1
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    
                    # Draw text background
                    cv2.rectangle(frame,
                                (x1, y2),
                                (x1 + text_size[0] + 10, y2 + text_size[1] + 10),
                                (0, 255, 0), cv2.FILLED)
                    
                    # Draw text
                    cv2.putText(frame, text,
                              (x1 + 5, y2 + text_size[1] + 5),
                              font, font_scale,
                              (255, 255, 255),
                              thickness)
                else:
                    # Unknown face
                    self.current_face_data = {
                        'is_unknown': True,
                        'bbox': bbox,
                        'is_fake': is_fake
                    }
                    
                    self.draw_unknown_face_box(frame, bbox)
                
        except Exception as e:
            print(f"Error processing face: {str(e)}")

    def calculate_confidence(self, similarity_score):
        """Calculate confidence score from similarity"""
        base_threshold = 0.5
        high_confidence_threshold = 0.7
        
        if similarity_score < base_threshold:
            return 0
        elif similarity_score > high_confidence_threshold:
            # Boost high confidence matches
            return 95 + (similarity_score - high_confidence_threshold) * 50
        else:
            # Linear scaling between thresholds
            return (similarity_score - base_threshold) * (95 / (high_confidence_threshold - base_threshold))
        
    def _cleanup_old_detections(self):
        """Remove detections older than 10 seconds"""
        current_time = time.time()
        self.last_detections = {
            k: v for k, v in self.last_detections.items()
            if current_time - v['timestamp'] < 10
        }

if __name__ == "__main__":
    app = AttendanceSystem()
    app.run()
