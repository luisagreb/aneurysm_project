"""
DICOM Angiography Viewer
- Load DICOM files with multiple frames
- Navigate through frames using slider or buttons
- Check pixel intensity with mouse hover/click
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

try:
    import pydicom
except ImportError:
    print("Please install pydicom: pip install pydicom")
    exit(1)

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Please install Pillow: pip install Pillow")
    exit(1)


class DicomViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("DICOM Angiography Viewer")
        self.root.geometry("1000x800")

        # Data variables
        self.dicom_data = None
        self.pixel_array = None
        self.current_frame = 0
        self.total_frames = 0
        self.display_image = None
        self.tk_image = None
        self.zoom_factor = 1.0

        # Window/Level for contrast adjustment
        self.window_center = 128
        self.window_width = 256

        self.setup_ui()

    def setup_ui(self):
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open DICOM...", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Top control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Open button
        ttk.Button(control_frame, text="Open DICOM File", command=self.open_file).pack(side=tk.LEFT, padx=5, pady=5)

        # Frame navigation
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(side=tk.LEFT, padx=20, pady=5)

        ttk.Button(nav_frame, text="<<", command=self.first_frame, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="<", command=self.prev_frame, width=3).pack(side=tk.LEFT, padx=2)

        self.frame_label = ttk.Label(nav_frame, text="Frame: 0/0", width=15)
        self.frame_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(nav_frame, text=">", command=self.next_frame, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text=">>", command=self.last_frame, width=3).pack(side=tk.LEFT, padx=2)

        # Frame slider
        slider_frame = ttk.Frame(control_frame)
        slider_frame.pack(side=tk.LEFT, padx=20, pady=5, fill=tk.X, expand=True)

        ttk.Label(slider_frame, text="Frame:").pack(side=tk.LEFT)
        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Window/Level controls
        wl_frame = ttk.LabelFrame(main_frame, text="Window/Level (Contrast)")
        wl_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(wl_frame, text="Window Center:").pack(side=tk.LEFT, padx=5)
        self.wc_var = tk.IntVar(value=128)
        self.wc_slider = ttk.Scale(wl_frame, from_=0, to=4095, orient=tk.HORIZONTAL,
                                    variable=self.wc_var, command=self.on_wl_change)
        self.wc_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.wc_label = ttk.Label(wl_frame, text="128", width=6)
        self.wc_label.pack(side=tk.LEFT, padx=5)

        ttk.Label(wl_frame, text="Window Width:").pack(side=tk.LEFT, padx=5)
        self.ww_var = tk.IntVar(value=256)
        self.ww_slider = ttk.Scale(wl_frame, from_=1, to=4095, orient=tk.HORIZONTAL,
                                    variable=self.ww_var, command=self.on_wl_change)
        self.ww_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.ww_label = ttk.Label(wl_frame, text="256", width=6)
        self.ww_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(wl_frame, text="Auto W/L", command=self.auto_window_level).pack(side=tk.LEFT, padx=10)

        # Image display area with scrollbars
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas for image
        self.canvas = tk.Canvas(image_frame, bg='black', cursor='crosshair')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<Button-1>', self.on_mouse_click)
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)  # Windows
        self.canvas.bind('<Button-4>', self.on_mouse_wheel)    # Linux scroll up
        self.canvas.bind('<Button-5>', self.on_mouse_wheel)    # Linux scroll down

        # Info panel
        info_frame = ttk.LabelFrame(main_frame, text="Information")
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        # Mouse position and intensity
        mouse_frame = ttk.Frame(info_frame)
        mouse_frame.pack(fill=tk.X, padx=5, pady=5)

        self.pos_label = ttk.Label(mouse_frame, text="Position: (-, -)", width=25)
        self.pos_label.pack(side=tk.LEFT, padx=10)

        self.intensity_label = ttk.Label(mouse_frame, text="Intensity: -", width=20)
        self.intensity_label.pack(side=tk.LEFT, padx=10)

        self.click_intensity_label = ttk.Label(mouse_frame, text="Clicked Intensity: -", width=25)
        self.click_intensity_label.pack(side=tk.LEFT, padx=10)

        # DICOM info
        self.dicom_info_label = ttk.Label(info_frame, text="No file loaded", wraplength=900)
        self.dicom_info_label.pack(fill=tk.X, padx=5, pady=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready. Open a DICOM file to begin.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('<Home>', lambda e: self.first_frame())
        self.root.bind('<End>', lambda e: self.last_frame())
        self.root.bind('<Control-o>', lambda e: self.open_file())

    def open_file(self):
        filetypes = [
            ("DICOM files", "*.dcm *.DCM *.dicom *.DICOM"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(title="Open DICOM File", filetypes=filetypes)

        if filepath:
            self.load_dicom(filepath)

    def load_dicom(self, filepath):
        try:
            self.status_var.set(f"Loading: {filepath}")
            self.root.update()

            # Read DICOM file
            self.dicom_data = pydicom.dcmread(filepath)
            self.pixel_array = self.dicom_data.pixel_array

            # Handle different array shapes
            if len(self.pixel_array.shape) == 2:
                # Single frame
                self.pixel_array = np.expand_dims(self.pixel_array, axis=0)
                self.total_frames = 1
            elif len(self.pixel_array.shape) == 3:
                # Multiple frames (frames, height, width) or RGB
                if self.pixel_array.shape[2] == 3:
                    # RGB image - convert to grayscale
                    self.pixel_array = np.mean(self.pixel_array, axis=2)
                    self.pixel_array = np.expand_dims(self.pixel_array, axis=0)
                    self.total_frames = 1
                else:
                    self.total_frames = self.pixel_array.shape[0]
            elif len(self.pixel_array.shape) == 4:
                # (frames, height, width, channels) - RGB video
                self.pixel_array = np.mean(self.pixel_array, axis=3)
                self.total_frames = self.pixel_array.shape[0]

            self.current_frame = 0

            # Update slider
            self.frame_slider.config(from_=0, to=max(0, self.total_frames - 1))
            self.frame_slider.set(0)

            # Auto window/level
            self.auto_window_level()

            # Update DICOM info
            info_text = self.get_dicom_info()
            self.dicom_info_label.config(text=info_text)

            # Display first frame
            self.display_frame()

            self.status_var.set(f"Loaded: {filepath} ({self.total_frames} frames)")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load DICOM file:\n{str(e)}")
            self.status_var.set("Error loading file")

    def get_dicom_info(self):
        if self.dicom_data is None:
            return "No file loaded"

        info_parts = []

        # Patient info
        if hasattr(self.dicom_data, 'PatientName'):
            info_parts.append(f"Patient: {self.dicom_data.PatientName}")
        if hasattr(self.dicom_data, 'PatientID'):
            info_parts.append(f"ID: {self.dicom_data.PatientID}")

        # Study info
        if hasattr(self.dicom_data, 'StudyDescription'):
            info_parts.append(f"Study: {self.dicom_data.StudyDescription}")
        if hasattr(self.dicom_data, 'SeriesDescription'):
            info_parts.append(f"Series: {self.dicom_data.SeriesDescription}")

        # Image info
        if hasattr(self.dicom_data, 'Rows') and hasattr(self.dicom_data, 'Columns'):
            info_parts.append(f"Size: {self.dicom_data.Columns}x{self.dicom_data.Rows}")
        info_parts.append(f"Frames: {self.total_frames}")

        if hasattr(self.dicom_data, 'BitsStored'):
            info_parts.append(f"Bits: {self.dicom_data.BitsStored}")

        return " | ".join(info_parts)

    def auto_window_level(self):
        if self.pixel_array is None:
            return

        frame = self.pixel_array[self.current_frame]
        self.window_center = int(np.mean(frame))
        self.window_width = int(np.std(frame) * 4)

        if self.window_width < 1:
            self.window_width = 256

        self.wc_var.set(self.window_center)
        self.ww_var.set(self.window_width)
        self.wc_label.config(text=str(self.window_center))
        self.ww_label.config(text=str(self.window_width))

        self.display_frame()

    def apply_window_level(self, image):
        """Apply window/level to image for display"""
        wc = self.window_center
        ww = self.window_width

        min_val = wc - ww / 2
        max_val = wc + ww / 2

        # Normalize to 0-255
        image = np.clip(image, min_val, max_val)
        image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        return image

    def display_frame(self):
        if self.pixel_array is None:
            return

        # Get current frame
        frame = self.pixel_array[self.current_frame].astype(np.float64)

        # Apply window/level
        display_array = self.apply_window_level(frame)

        # Store for intensity lookup
        self.current_raw_frame = self.pixel_array[self.current_frame]

        # Create PIL image
        self.display_image = Image.fromarray(display_array)

        # Resize to fit canvas while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            img_width, img_height = self.display_image.size

            # Calculate scale to fit
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale = min(scale_w, scale_h) * self.zoom_factor

            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            if new_width > 0 and new_height > 0:
                resized = self.display_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.tk_image = ImageTk.PhotoImage(resized)

                # Store scale and offset for coordinate conversion
                self.display_scale = scale
                self.display_offset_x = (canvas_width - new_width) // 2
                self.display_offset_y = (canvas_height - new_height) // 2

                # Display on canvas
                self.canvas.delete("all")
                self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                        image=self.tk_image, anchor=tk.CENTER)

        # Update frame label
        self.frame_label.config(text=f"Frame: {self.current_frame + 1}/{self.total_frames}")

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image coordinates"""
        if not hasattr(self, 'display_scale'):
            return None, None

        img_x = (canvas_x - self.display_offset_x) / self.display_scale
        img_y = (canvas_y - self.display_offset_y) / self.display_scale

        # Check bounds
        if self.pixel_array is not None:
            height, width = self.pixel_array[self.current_frame].shape
            if 0 <= img_x < width and 0 <= img_y < height:
                return int(img_x), int(img_y)

        return None, None

    def on_mouse_move(self, event):
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)

        if img_x is not None and img_y is not None:
            self.pos_label.config(text=f"Position: ({img_x}, {img_y})")

            if hasattr(self, 'current_raw_frame'):
                intensity = self.current_raw_frame[img_y, img_x]
                self.intensity_label.config(text=f"Intensity: {intensity}")
        else:
            self.pos_label.config(text="Position: (-, -)")
            self.intensity_label.config(text="Intensity: -")

    def on_mouse_click(self, event):
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)

        if img_x is not None and img_y is not None:
            if hasattr(self, 'current_raw_frame'):
                intensity = self.current_raw_frame[img_y, img_x]
                self.click_intensity_label.config(text=f"Clicked: ({img_x}, {img_y}) = {intensity}")

    def on_mouse_wheel(self, event):
        # Scroll through frames with mouse wheel
        if self.pixel_array is None:
            return

        if event.num == 5 or event.delta < 0:
            self.next_frame()
        elif event.num == 4 or event.delta > 0:
            self.prev_frame()

    def on_slider_change(self, value):
        new_frame = int(float(value))
        if new_frame != self.current_frame:
            self.current_frame = new_frame
            self.display_frame()

    def on_wl_change(self, value=None):
        self.window_center = self.wc_var.get()
        self.window_width = self.ww_var.get()
        self.wc_label.config(text=str(self.window_center))
        self.ww_label.config(text=str(self.window_width))
        self.display_frame()

    def first_frame(self):
        if self.pixel_array is not None:
            self.current_frame = 0
            self.frame_slider.set(0)
            self.display_frame()

    def last_frame(self):
        if self.pixel_array is not None:
            self.current_frame = self.total_frames - 1
            self.frame_slider.set(self.current_frame)
            self.display_frame()

    def prev_frame(self):
        if self.pixel_array is not None and self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.set(self.current_frame)
            self.display_frame()

    def next_frame(self):
        if self.pixel_array is not None and self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_slider.set(self.current_frame)
            self.display_frame()


def main():
    root = tk.Tk()
    app = DicomViewer(root)

    # Handle window resize
    def on_resize(event):
        if app.pixel_array is not None:
            app.display_frame()

    root.bind('<Configure>', on_resize)
    root.mainloop()


if __name__ == "__main__":
    main()
