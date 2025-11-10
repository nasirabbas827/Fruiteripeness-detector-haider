import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2 # OpenCV for image resizing

# --- 1. CONFIGURATION (Must match your training setup) ---
MODEL_PATH = "tomato_ripeness_model.h5"
IMG_SIZE = (128, 128)
CLASS_NAMES = ["unripe", "ripe", "overripe"]

# --- 2. MODEL LOADING ---
try:
    # Load the trained Keras model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Successfully loaded model from: {MODEL_PATH}")
except Exception as e:
    messagebox.showerror("Model Error", f"Error loading model: {e}\nPlease ensure '{MODEL_PATH}' is in the same directory.")
    model = None

# --- 3. PREDICTION FUNCTION ---
def predict_ripeness(image_path):
    """Loads, preprocesses, and predicts the ripeness of an image."""
    if model is None:
        return "Model Error", 0.0

    try:
        # Load image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return "File Error", 0.0
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocessing (Resize and Normalize)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_normalized = img_resized / 255.0
        
        # Add batch dimension (1, 128, 128, 3)
        img_input = np.expand_dims(img_normalized, axis=0)

        # Predict class
        preds = model.predict(img_input, verbose=0)
        pred_class_index = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))

        pred_label = CLASS_NAMES[pred_class_index]
        return pred_label, confidence

    except Exception as e:
        return f"Prediction Error", 0.0


# --- 4. GUI UTILITIES ---

def exit_app():
    if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
        root.destroy()

def clear_display():
    # Clear the image
    result_image_label.config(image='')
    result_image_label.image = None
    # Clear the result text
    result_text_var.set("Prediction will appear here after image selection.")
    result_label.config(background='#ffffff')


def create_button(parent, text, command, height, width, color):
    """Utility function to create styled buttons in the sidebar."""
    button = tk.Button(parent, text=text, command=command, height=height, width=width, 
                       bg=color, fg='#000000', # Black foreground text
                       font=('Arial', 12, 'bold'), 
                       activebackground=color, borderwidth=0)
    button.pack(pady=10)
    return button

# --- 5. CORE IMAGE PROCESSING & PREDICTION ---
def open_image_and_predict():
    """Handles image upload, prediction, and GUI update."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")],
        title="Select a Fruit Image"
    )
    if not file_path:
        return

    # Reset text display
    result_text_var.set("Processing image...")
    
    # --- A. Display Image in GUI ---
    try:
        img = Image.open(file_path)
        
        # Resize image to fit the display frame, max 450x450
        max_size = (450, 450)
        img.thumbnail(max_size)
        
        # Convert PIL image to Tkinter-compatible image
        global result_img_tk 
        result_img_tk = ImageTk.PhotoImage(img)
        result_image_label.config(image=result_img_tk, text="", relief="flat")
        result_image_label.image = result_img_tk
        
    except Exception as e:
        messagebox.showerror("Image Load Error", f"Could not load image: {e}")
        clear_display()
        return

    # --- B. Run Prediction ---
    pred_label, confidence = predict_ripeness(file_path)

    # --- C. Update Result Label and Color Coding ---
    if "Error" in pred_label:
        final_text = f"An error occurred during prediction."
        bg_color = '#FFD2C8' # Light Red/Error
    else:
        # Determine background color based on prediction
        if pred_label == "unripe":
            bg_color = '#B9FBC0' # Light Green
        elif pred_label == "ripe":
            bg_color = '#FFC38A' # Light Orange/Yellow
        elif pred_label == "overripe":
            bg_color = '#FFA07A' # Light Red/Brown
        else:
            bg_color = '#ffffff' # Default white

        final_text = (
            f"Classification: {pred_label.upper()}\n"
            f"Confidence: {confidence*100:.2f}%"
        )
    
    result_text_var.set(final_text)
    result_label.config(background=bg_color)


# --- 6. TKINTER GUI SETUP ---
root = tk.Tk()
root.title("🍅 Fruit Ripeness Detection (CNN)")
root.geometry("900x600")

# Center the window
window_width = 900
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
root.geometry(f'{window_width}x{window_height}+{x}+{y}')

# Style configuration
style = ttk.Style()
style.configure('MainFrame.TFrame', background='#e1f5fe')  # Light Blue Main Background
style.configure('SidebarFrame.TFrame', background='#e0f7fa') # Lighter Blue Sidebar
style.configure('ResultFrame.TFrame', background='#ffffff') # White Result Background
style.configure('Title.TLabel', font=('Arial', 18, 'bold'), background='#e0f7fa', foreground='#000000')


# Main frame (fills the whole window)
main_frame = ttk.Frame(root, padding="15", style='MainFrame.TFrame')
main_frame.pack(fill=tk.BOTH, expand=True)

# Sidebar frame (for controls)
sidebar_frame = ttk.Frame(main_frame, width=200, padding="20", style='SidebarFrame.TFrame', relief="raised")
sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))

# Title in Sidebar
title_label = ttk.Label(sidebar_frame, text="Ripeness\nDetector", style='Title.TLabel', anchor='center')
title_label.pack(pady=(0, 30), fill=tk.X)


# Result display frame (for image and text)
result_frame = ttk.Frame(main_frame, padding="20", style='ResultFrame.TFrame', relief="sunken")
result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# --- Result Frame Layout ---
# Image Display Label (Takes up most of the space)
result_image_label = ttk.Label(result_frame, background='#ffffff', relief="solid", 
                               text="Upload an image (JPG/PNG)", anchor='center')
result_image_label.pack(expand=True, fill=tk.BOTH, pady=(0, 15))

# Prediction Text Label (Below the image)
result_text_var = tk.StringVar()
result_text_var.set("Prediction will appear here after image selection.")

result_label = tk.Label(result_frame, textvariable=result_text_var, 
                        font=('Arial', 16, 'bold'), 
                        background='#ffffff',  # Default white background
                        relief="ridge", justify=tk.CENTER, pady=10)
result_label.pack(fill=tk.X)


# --- Sidebar Buttons ---
# Create buttons with specified height, width, and colors
create_button(sidebar_frame, "Select Image", open_image_and_predict, 3, 15, "#66C2FF")
create_button(sidebar_frame, "Clear Display", clear_display, 3, 15, "#FFA07A")
create_button(sidebar_frame, "Exit Application", exit_app, 3, 15, "#CD5C5C")

# Start the application
if __name__ == '__main__':
    root.mainloop()