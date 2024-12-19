import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from theme_utils import ThemeManager

class GUI:
    def __init__(self, app):
        self.app = app  # Reference to main AttendanceSystem instance
        self.root = app.root
        self.colors = app.colors
        self.setup_gui()

    def create_rounded_rect(self, canvas, x1, y1, x2, y2, radius=25, **kwargs):
        """Create a rounded rectangle"""
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)

    def setup_gui(self):
        """Setup all GUI components"""
        self.setup_theme_toggle()
        self.setup_device_frame()
        self.setup_mode_frame()
        self.setup_controls()
        self.setup_status()
        self.setup_video_display()
        self.update_colors()

    def setup_theme_toggle(self):
        """Setup theme toggle switch"""
        theme_frame = tk.Frame(self.root)
        theme_frame.pack(fill="x", padx=10, pady=5)
        
        toggle_frame = tk.Frame(theme_frame, width=60, height=30)
        toggle_frame.pack(side=tk.RIGHT, padx=10)
        
        self.toggle_canvas = tk.Canvas(
            toggle_frame, 
            width=60, height=30,
            bg=self.colors['light']['bg'],
            highlightthickness=0
        )
        self.toggle_canvas.pack()
        
        # Store reference to GUI instance in canvas
        self.toggle_canvas.gui = self
        
        self.draw_toggle_switch()
        self.toggle_canvas.bind('<Button-1>', self.app.toggle_theme)
        
        tk.Label(
            theme_frame,
            text="Dark Mode",
            font=('Arial', 10)
        ).pack(side=tk.RIGHT)

    def setup_device_frame(self):
        """Setup device selection frame"""
        device_frame = tk.LabelFrame(self.root, text="Device", padx=10, pady=5)
        device_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Radiobutton(
            device_frame,
            text="GPU (Recommended)",
            variable=self.app.device_var,
            value="GPU",
            command=self.app.change_device
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Radiobutton(
            device_frame,
            text="CPU",
            variable=self.app.device_var,
            value="CPU",
            command=self.app.change_device
        ).pack(side=tk.LEFT, padx=5)

    def setup_mode_frame(self):
        """Setup mode selection frame"""
        mode_frame = tk.LabelFrame(self.root, text="Mode", padx=10, pady=5)
        mode_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(
            mode_frame,
            text="Re-train",
            command=self.app.train_mode,
            bg=self.colors['light']['button_bg']['warning'],
            fg='white'
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            mode_frame,
            text="Register New Face",
            command=self.app.start_registration,
            bg=self.colors['light']['button_bg']['purple'],
            fg='white'
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            mode_frame,
            text="Check Image",
            command=self.app.check_image,
            bg=self.colors['light']['button_bg']['indigo'],
            fg='white'
        ).pack(side=tk.LEFT, padx=5)
        
        # Add Check Frame button
        tk.Button(
            mode_frame,
            text="Check Frame",
            command=self.app.ui_handlers.check_frame,
            bg=self.colors['light']['button_bg']['primary'],
            fg='white'
        ).pack(side=tk.LEFT, padx=5)

    def setup_controls(self):
        """Setup control frame with buttons"""
        self.control_frame = tk.LabelFrame(self.root, text="Controls", padx=10, pady=5)
        self.control_frame.pack(fill="x", padx=10, pady=5)
        
        # Student Selection Frame
        selection_frame = tk.Frame(self.control_frame)
        selection_frame.pack(side=tk.LEFT, padx=5)
        
        tk.Label(selection_frame, text="Select Student ID:").pack(side=tk.LEFT, padx=5)
        
        self.student_selector = ttk.Combobox(
            selection_frame,
            state='readonly',
            width=20
        )
        self.student_selector.pack(side=tk.LEFT, padx=5)
        
        # Improved refresh button
        refresh_btn = tk.Button(
            selection_frame,
            text="↻",
            command=self.app.ui_handlers.update_student_list,
            bg=self.colors['light']['button_bg']['primary'],
            fg='white',
            font=('Arial', 12, 'bold'),
            width=2,
            height=1,
            relief=tk.FLAT,
            borderwidth=0,
            padx=5,
            pady=0
        )
        refresh_btn.pack(side=tk.LEFT, padx=(2, 10))
        
        # Recognition toggle button
        self.toggle_btn = tk.Button(
            self.control_frame,
            text="Start Recognition",
            command=self.app.ui_handlers.toggle_recognition,
            bg=self.colors['light']['button_bg']['primary'],
            fg='white'
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=5)
        
        # Add to Database button
        self.add_db_btn = tk.Button(
            self.control_frame,
            text="Add to Database",
            command=self.app.add_to_database,
            bg=self.colors['light']['button_bg']['success'],
            fg='white',
            state='normal'
        )
        
        # Mismatched Face button
        self.mismatch_btn = tk.Button(
            self.control_frame,
            text="Mismatched Face",
            command=self.app.handle_mismatch,
            bg=self.colors['light']['button_bg']['danger'],
            fg='white',
            state='normal'
        )
        
        # Quit button
        tk.Button(
            self.control_frame,
            text="Quit",
            command=self.app.quit_application,
            bg=self.colors['light']['button_bg']['danger'],
            fg='white',
            font=('Arial', 11, 'bold')
        ).pack(side=tk.RIGHT, padx=10)

    def setup_status(self):
        """Setup status label"""
        self.status_label = tk.Label(
            self.root, 
            text="System Ready",
            font=('Arial', 10)
        )
        self.status_label.pack(pady=5)

    def setup_video_display(self):
        """Setup video display area"""
        self.video_label = tk.Label(self.root)
        self.video_label.pack(pady=10)

    def clear_video_display(self):
        """Clear the video display"""
        # Create a blank image of the same size as the video display
        width = 640  # Default width
        height = 480  # Default height
        
        # Create a black image
        blank_image = Image.new('RGB', (width, height), color='black')
        blank_photo = ImageTk.PhotoImage(blank_image)
        
        # Update label
        self.video_label.configure(image=blank_photo)
        self.video_label.image = blank_photo  # Keep a reference

    def draw_toggle_switch(self):
        """Draw theme toggle switch"""
        self.toggle_canvas.delete("all")
        
        # Get current theme colors
        theme = 'dark' if self.app.dark_mode.get() else 'light'
        bg_color = self.colors[theme]['bg']
        switch_color = self.colors[theme]['button_bg']['primary']
        
        # Create rounded rectangle directly
        points = [
            5+10, 5,      # Top left with radius
            55-10, 5,     # Top right with radius
            55, 5,
            55, 5+10,
            55, 25-10,
            55, 25,
            55-10, 25,    # Bottom right with radius
            5+10, 25,     # Bottom left with radius
            5, 25,
            5, 25-10,
            5, 5+10,
            5, 5
        ]
        self.toggle_canvas.create_polygon(
            points, 
            smooth=True,
            fill=switch_color if self.app.dark_mode.get() else '#ccc'
        )
        
        # Draw switch circle
        circle_x = 35 if self.app.dark_mode.get() else 15
        self.toggle_canvas.create_oval(
            circle_x, 7,
            circle_x + 16, 23,
            fill='white',
            outline='white'
        )

    def update_colors(self):
        """Update colors for all widgets"""
        theme = 'dark' if self.app.dark_mode.get() else 'light'
        colors = self.colors[theme]
        
        # Update root background
        self.root.configure(bg=colors['bg'])
        
        # Helper function to update widget colors
        def update_widget_colors(widget):
            try:
                if isinstance(widget, (tk.Frame, tk.LabelFrame)):
                    widget.configure(bg=colors['bg'])
                    if isinstance(widget, tk.LabelFrame):
                        widget.configure(fg=colors['fg'])
                elif isinstance(widget, tk.Label):
                    widget.configure(bg=colors['bg'], fg=colors['fg'])
                elif isinstance(widget, tk.Button):
                    # Preserve button colors based on text/function
                    text = str(widget['text']).lower()
                    if 'quit' in text:
                        new_bg = colors['button_bg']['danger']
                    elif 'add' in text:
                        new_bg = colors['button_bg']['success']
                    elif 're-train' in text:
                        new_bg = colors['button_bg']['warning']
                    elif 'register' in text:
                        new_bg = colors['button_bg']['purple']
                    elif 'check' in text:
                        new_bg = colors['button_bg']['indigo']
                    elif 'mismatch' in text:
                        new_bg = colors['button_bg']['danger']
                    elif 'stop' in text:
                        new_bg = colors['button_bg']['danger']
                    else:
                        new_bg = colors['button_bg']['primary']
                    
                    widget.configure(
                        bg=new_bg,
                        fg=colors['button_fg'],
                        activebackground=ThemeManager.adjust_color_brightness(new_bg, 1.1),
                        activeforeground=colors['button_fg']
                    )
                elif isinstance(widget, ttk.Combobox):
                    style = ttk.Style()
                    style.configure('TCombobox',
                                  fieldbackground=colors['input_bg'],
                                  background=colors['bg'],
                                  foreground=colors['input_fg'])
                    widget.configure(style='TCombobox')
                
                # Update children widgets
                for child in widget.winfo_children():
                    update_widget_colors(child)
                    
            except Exception as e:
                if "unknown option \"-fg\"" not in str(e):
                    print(f"Error updating colors for widget {widget}: {e}")
        
        # Update all widgets
        update_widget_colors(self.root) 