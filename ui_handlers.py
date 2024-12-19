from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import tkinter.messagebox as messagebox

class UIEventHandlers:
    def __init__(self, app):
        self.app = app

    def update_student_list(self):
        """Update the list of available students"""
        base_path = Path("assets/student_images")
        if base_path.exists():
            # Get list of student IDs (folder names)
            student_ids = [folder.name for folder in base_path.iterdir() if folder.is_dir()]
            student_ids.sort()  # Sort IDs
            
            # Update combobox values
            self.app.gui.student_selector.configure(values=['All'] + student_ids)
            
            # Set to 'All' by default
            self.app.gui.student_selector.set('All')
            
            print(f"Found student IDs: {', '.join(student_ids)}")
        else:
            print("Student images directory not found")

    def toggle_theme(self, event=None):
        """Handle theme toggle"""
        self.app.dark_mode.set(not self.app.dark_mode.get())
        self.app.gui.draw_toggle_switch()
        self.app.gui.update_colors()

    def toggle_recognition(self):
        """Handle recognition toggle"""
        try:
            self.app.is_recognition_active = not self.app.is_recognition_active
            if self.app.is_recognition_active:
                if self.app.cap is None or not self.app.cap.isOpened():
                    self.app.cap = self.app.init_camera()
                    if self.app.cap is None:
                        return
                self.app.gui.toggle_btn.config(text="Stop Recognition", bg='#f44336')
                # Show database operation buttons
                self.app.gui.add_db_btn.pack(side='left', padx=5)
                self.app.gui.mismatch_btn.pack(side='left', padx=5)
                self.app.update_video_feed()
            else:
                self.app.gui.toggle_btn.config(text="Start Recognition", bg='#2196F3')
                # Hide database operation buttons
                self.app.gui.add_db_btn.pack_forget()
                self.app.gui.mismatch_btn.pack_forget()
                # Ensure camera is released when stopping
                if self.app.cap is not None:
                    self.app.cap.release()
                    self.app.cap = None
                # Clear the video display
                self.app.gui.clear_video_display()
        except Exception as e:
            print(f"Error toggling recognition: {e}")
            self.app.is_recognition_active = False
            self.app.gui.toggle_btn.config(text="Start Recognition", bg='#2196F3')
            # Hide database operation buttons
            self.app.gui.add_db_btn.pack_forget()
            self.app.gui.mismatch_btn.pack_forget()
            if self.app.cap is not None:
                self.app.cap.release()
                self.app.cap = None
            # Clear the video display
            self.app.gui.clear_video_display()

    def check_frame(self):
        """Handle check frame functionality"""
        # Create preview window
        preview_window = tk.Toplevel(self.app.root)
        preview_window.title("Camera Preview")
        preview_window.geometry("800x700")
        
        # Create video display
        video_label = tk.Label(preview_window)
        video_label.pack(pady=10)
        
        # Create button frame
        button_frame = tk.Frame(preview_window)
        button_frame.pack(pady=5)
        
        # Initialize camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not access camera!")
            preview_window.destroy()
            return
        
        def update_preview():
            """Update camera preview"""
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Convert frame for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    video_label.imgtk = imgtk
                    video_label.configure(image=imgtk)
                preview_window.after(10, update_preview)
        
        def capture_and_process():
            """Capture and process current frame"""
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Release camera and close preview window
                    cap.release()
                    preview_window.destroy()
                    
                    # Process frame
                    process_frame(frame)
        
        def process_frame(frame):
            """Process captured frame and display results"""
            try:
                # Create result window
                result_window = tk.Toplevel(self.app.root)
                result_window.title("Frame Analysis")
                
                # Check for spoofing
                is_real, spoof_confidence, face_results, processed_frame = self.app.spoof_detector.predict(frame)
                
                if face_results is not None:
                    faces = self.app.face_analyzer.get(frame)
                    
                    # Process each face
                    for face, result in zip(faces, face_results):
                        bbox = result['bbox']
                        is_fake = not result['is_real']
                        
                        # Process face recognition
                        self.app.process_face(face, processed_frame, 'All', is_fake=is_fake)
                
                # Convert and display result
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Resize if needed
                max_size = (800, 600)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                imgtk = ImageTk.PhotoImage(img)
                
                # Display result
                label = tk.Label(result_window, image=imgtk)
                label.image = imgtk
                label.pack(pady=10)
                
                # Add close button
                tk.Button(
                    result_window,
                    text="Close",
                    command=result_window.destroy,
                    bg='#f44336',
                    fg='white'
                ).pack(pady=5)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error processing frame: {str(e)}")
        
        def on_window_close():
            """Handle window closing"""
            if cap.isOpened():
                cap.release()
            preview_window.destroy()
        
        # Create capture button
        tk.Button(
            button_frame,
            text="Capture Frame",
            command=capture_and_process,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 11, 'bold')
        ).pack(side=tk.LEFT, padx=5)
        
        # Create cancel button
        tk.Button(
            button_frame,
            text="Cancel",
            command=on_window_close,
            bg='#f44336',
            fg='white',
            font=('Arial', 11)
        ).pack(side=tk.LEFT, padx=5)
        
        # Set window close handler
        preview_window.protocol("WM_DELETE_WINDOW", on_window_close)
        
        # Start preview
        update_preview()