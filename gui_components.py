import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

class GUI:
    def __init__(self, app):
        self.app = app  # Reference to main AttendanceSystem instance
        self.root = app.root
        
        self.is_fullscreen = False
        self.is_dark_theme = os.environ.get('APP_THEME', 'light') == 'dark'
        
        self.setup_menubar()
        self.setup_gui()
    
    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes('-fullscreen', self.is_fullscreen)
        
    
    def toggle_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        os.environ['APP_THEME'] = 'dark' if self.is_dark_theme else 'light'
        self.apply_theme()
        
    def apply_theme(self):
        if self.is_dark_theme:
            self.root.tk_setPalette(background='black', foreground='white')
        else:
            self.root.tk_setPalette(background='white', foreground='black')

    def setup_menubar(self):
        """Setup application menubar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Re-train System", command=self.app.train_mode, accelerator="Command-t")
        file_menu.add_command(label="Register New Face", command=self.app.start_registration, accelerator="Command-r")
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.app.quit_application, accelerator="Command-q")
        
        # device Menu
        device_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Device", menu=device_menu)
        
        device_menu.add_radiobutton(label="GPU (Recommended)", variable=self.app.device_var, value="GPU", command=self.app.change_device)
        device_menu.add_radiobutton(label="CPU", variable=self.app.device_var, value="CPU", command=self.app.change_device)
        
        device_menu.add_separator()
        text = "Current Device: " + self.app.device.upper()
        # display the current device
        device_menu.add_command(label=text, state='disabled')
        
        # Recognition Menu
        recog_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Recognition", menu=recog_menu)
        recog_menu.add_command(label="Check Image", command=self.app.check_image, accelerator="Command-i")
        recog_menu.add_command(label="Check Frame", command=self.app.ui_handlers.check_frame, accelerator="Command-F")
        
        #View Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Fullscreen", command=self.toggle_fullscreen)
        view_menu.add_command(label="Toggle Theme", command=self.toggle_theme)
        
        # Input menu
        input_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Input", menu=input_menu)
        input_menu.add_command(label="Choose camera")
        
        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.open_docs)
        help_menu.add_command(label="About", command=self.show_about)
        
        
        # Add keyboard shortcuts
        self.root.bind('<Command-t>', lambda e: self.app.train_mode())
        self.root.bind('<Command-r>', lambda e: self.app.start_registration())
        self.root.bind('<Command-q>', lambda e: self.app.quit_application())
        self.root.bind('<Command-i>', lambda e: self.app.check_image())
        self.root.bind('<Command-f>', lambda e: self.app.ui_handlers.check_frame())

    # create open document function
    def open_docs(self):
        # open tkinter window
        self.docs_window = tk.Toplevel(self.root)
        self.docs_window.title("Documentation")
        self.docs_window.geometry("800x600")
        self.docs_window.configure(bg="white")
        self.docs_window.resizable(False, False)
        # create a text widget
        self.docs_text = tk.Text(self.docs_window, wrap=tk.WORD)
        self.docs_text.pack(fill="both", expand=True)
        # open the documentation file
        try:
            with open("docs.txt", "r") as f:
                self.docs_text.insert(tk.END, f.read())
        except Exception as e:
            # close the window if the file is not found
            self.docs_window.destroy()
            # show error message
            tk.messagebox.showerror("Error", "Documentation file not found.")
        # disable editing
        self.docs_text.config(state=tk.DISABLED)
        
    def show_about(self):
        # open tkinter window
        self.about_window = tk.Toplevel(self.root)
        self.about_window.title("About")
        self.about_window.geometry("400x300")
        self.about_window.configure(bg="white")
        self.about_window.resizable(False, False)
        # create a text widget
        self.about_text = tk.Text(self.about_window, wrap=tk.WORD)
        self.about_text.pack(fill="both", expand=True)
        # open the documentation file
        with open("about.txt", "r") as f:
            self.about_text.insert(tk.END, f.read())
        # disable editing
        self.about_text.config(state=tk.DISABLED)
                        
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

        self.setup_controls()
        self.setup_status()
        self.setup_video_display()

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
            font=('Arial', 12, 'bold'),
            width=2,
            height=1,
            relief=tk.FLAT,
            borderwidth=0,
        )
        refresh_btn.pack(side=tk.LEFT, padx=(2, 10))
        
        # Recognition toggle button
        self.toggle_btn = tk.Button(
            self.control_frame,
            text="Start Recognition",
            command=self.app.ui_handlers.toggle_recognition,
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=5)
        
        # Add to Database button
        self.add_db_btn = tk.Button(
            self.control_frame,
            text="Add to Database",
            command=self.app.add_to_database,
            state='normal'
        )
        
        # Mismatched Face button
        self.mismatch_btn = tk.Button(
            self.control_frame,
            text="Mismatched Face",
            command=self.app.handle_mismatch,

            state='normal'
        )
        
        # Quit button
        tk.Button(
            self.control_frame,
            text="Quit",
            command=self.app.quit_application,
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

