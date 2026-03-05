import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import pickle
from sklearn.preprocessing import StandardScaler

# Load model, scaler, and label map
model = pickle.load(open("fly_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_map = pickle.load(open("label_map.pkl", "rb"))
reverse_label_map = {v: k for k, v in label_map.items()}

feature_names = [
    "Area (µm²)", "Perimeter (µm)", "Circularity", "Aspect Ratio",
    "Contrast (GLCM)", "Homogeneity (GLCM)", "Energy (GLCM)",
    "LBP0", "LBP1", "LBP2", "LBP3", "LBP4", "LBP5", "LBP6", "LBP7", "LBP8", "LBP9"
]

def extract_features(image):
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    img_size = 128
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = gray[y:y + h, x:x + w]
            roi_resized = cv2.resize(roi, (img_size, img_size))
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            aspect_ratio = float(w) / h
            glcm = graycomatrix(roi_resized, distances=[1], angles=[0], levels=256,
                                symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            lbp = local_binary_pattern(roi_resized, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
            all_features = [area, perimeter, circularity, aspect_ratio,
                            contrast, homogeneity, energy] + list(lbp_hist)
            features.append(all_features)
    if not features:
        return np.zeros(34), []
    features = np.array(features)
    mean_feats = np.mean(features, axis=0)
    std_feats = np.std(features, axis=0)
    return np.concatenate([mean_feats, std_feats]), list(mean_feats.round(2))

# === GUI Setup ===
root = tk.Tk()
root.title("Anemia Detection using Image Processing")
root.geometry("1100x880")
root.configure(bg="#f0f4f8")

tk.Label(root, text="🔬 Detection of Different Types of Anemia using Image Processing Techniques",
         font=("Helvetica", 16, "bold"), bg="#f0f4f8", fg="#1a1a1a").pack(pady=10)

grid_frame = tk.Frame(root, bg="#f0f4f8")
grid_frame.pack(fill="both", expand=True, padx=10)
grid_frame.grid_rowconfigure((0, 1), weight=1)
grid_frame.grid_columnconfigure((0, 1, 2), weight=1)

# === Read Image Section ===
read_frame = tk.LabelFrame(grid_frame, text="📷 Read Image", font=("Helvetica", 11, "bold"),
                           bg="#e4ecf1", width=300, labelanchor="n")
read_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

img_canvas = tk.Canvas(read_frame, width=280, height=200, bg="white", bd=1, relief="sunken")
img_canvas.pack(pady=8)

select_btn = ttk.Button(read_frame, text="📂 Select Image")
select_btn.pack(pady=5)

predicted_label = tk.Label(read_frame, text="Predicted Anemia Type: --",
                           font=("Helvetica", 10, "bold"), bg="#e4ecf1", fg="darkred")
predicted_label.pack(pady=2)

confidence_label = tk.Label(read_frame, text="Confidence: --%", font=("Helvetica", 10),
                            bg="#e4ecf1", fg="#000080")
confidence_label.pack(pady=2)

# === Features Extracted Section ===
features_frame = tk.LabelFrame(grid_frame, text="🧬 Features Extracted", font=("Helvetica", 11, "bold"),
                               bg="#e4ecf1", labelanchor="n")
features_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

features_text = tk.Text(features_frame, height=19, width=44, font=("Courier New", 9), bd=1, relief="sunken")
features_text.pack(padx=8, pady=8)

# === Confusion Matrix ===
cm_frame = tk.LabelFrame(grid_frame, text="📈 Confusion Matrix", font=("Helvetica", 11, "bold"),
                         bg="#e4ecf1", labelanchor="n")
cm_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

cm_label = tk.Label(cm_frame, bg="white", bd=1, relief="sunken")
cm_label.pack(padx=8, pady=10)

# === Description Box ===
desc_frame = tk.LabelFrame(grid_frame, text="📝 Description", font=("Helvetica", 11, "bold"),
                           bg="#e4ecf1", labelanchor="n")
desc_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

desc_text = tk.Text(desc_frame, height=10, width=90, font=("Courier New", 9), bd=1, relief="sunken")
desc_text.pack(pady=10)

# === Classification Report ===
report_frame = tk.LabelFrame(grid_frame, text="📋 classification report.png", font=("Helvetica", 11, "bold"),
                             bg="#e4ecf1", labelanchor="n")
report_frame.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")

report_label = tk.Label(report_frame, bg="white", bd=1, relief="sunken")
report_label.pack(padx=8, pady=10)

# === Footer ===
footer_frame = tk.Frame(root, bg="#1a1a1a")
footer_frame.pack(fill="x")
tk.Label(
    footer_frame,
    text=(
        "Developed by: Tejaswini Kaladagi\n"
        "Department of MCA\n"
        "BLDEA's V.P. Dr. P. G. Halakatti College of Engineering & Technology, Vijayapur - 586103"
    ),
    font=("Helvetica", 11, "bold"),
    bg="#1a1a1a",
    fg="white",
    justify="center"
).pack(pady=10)


def analyze_image():
    try:
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path:
            return

        img_cv = cv2.imread(file_path)
        if img_cv is None:
            messagebox.showerror("Error", "Invalid image file.")
            return

        features, feature_list = extract_features(img_cv)
        if features.shape[0] != 34:
            messagebox.showerror("Error", "Failed to extract valid features.")
            return

        scaled = scaler.transform([features])
        probs = model.predict_proba(scaled)[0]
        pred = np.argmax(probs)
        confidence = probs[pred] * 100
        pred_class = reverse_label_map[pred]

        # Show selected image
        img_pil = Image.open(file_path).resize((280, 200), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        img_canvas.create_image(0, 0, anchor="nw", image=img_tk)
        img_canvas.image = img_tk

        # Display extracted features with units
        features_text.delete("1.0", tk.END)
        for name, val in zip(feature_names, feature_list):
            label, *unit = name.split(" ")
            unit_str = " ".join(unit)
            if unit_str:
                features_text.insert(tk.END, f"- {label}: {val} {unit_str}\n")
            else:
                features_text.insert(tk.END, f"- {name}: {val}\n")


        # Labels
        predicted_label.config(text=f"Predicted Anemia Type: {pred_class}")
        confidence_label.config(text=f"Confidence: {confidence:.2f}%")

      
        # Description
        desc_text.delete("1.0", tk.END)
        desc_text.insert(tk.END, f"🧪 Predicted Anemia Type: {pred_class.upper()}\n\n")
        desc_text.insert(tk.END, f"✅ Confidence Score: {confidence:.2f}%\n\n")
        desc_text.insert(tk.END, (
            "📖 Description:\n"
            "This result is based on detailed feature extraction from the blood smear image, "
            "analyzed using an ensemble model trained on various morphological and texture-based features. "
            "The model leverages techniques like contour analysis, GLCM texture properties, and LBP patterns.\n\n"
            "📊 For more insights, review the Confusion Matrix and the Classification Report displayed."
        ))


        # === Confusion Matrix Image ===
        try:
            cm_path = "confusion_matrix.png"
            if os.path.exists(cm_path):
                cm_img = Image.open(cm_path).resize((500, 260), Image.Resampling.LANCZOS)
                cm_tk = ImageTk.PhotoImage(cm_img)
                cm_label.configure(image=cm_tk)
                cm_label.image = cm_tk
            else:
                cm_label.config(text="confusion_matrix.png not found.", image="")
        except Exception as e:
            cm_label.config(text=f"Error displaying confusion matrix:\n{str(e)}", image="")

        # === Classification Report Image ===
     
        try:
            report_path = r"E:\anemia detection using ml\Classifiaction report.png"  # exact path and filename
            if os.path.exists(report_path):
                report_img = Image.open(report_path).resize((500, 260), Image.Resampling.LANCZOS)
                report_tk = ImageTk.PhotoImage(report_img)
                report_label.configure(image=report_tk)
                report_label.image = report_tk
            else:
                report_label.config(text="Classifiaction report.png not found.", image="")
        except Exception as e:
            report_label.config(text=f"Error displaying classification report:\n{str(e)}", image="")


    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong:\n{str(e)}")


select_btn.config(command=analyze_image)
root.mainloop()
