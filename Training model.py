import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle

# --- Config ---
image_dir = 'Labeled_Images1'
img_size = 128

# --- Feature Extraction (Fly ROI + Texture + Shape) ---
def extract_features(image):
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
        return np.zeros(34)

    features = np.array(features)
    mean_feats = np.mean(features, axis=0)
    std_feats = np.std(features, axis=0)
    return np.concatenate([mean_feats, std_feats])  # 17 mean + 17 std = 34

# --- Load Dataset and Extract Features ---
print("\U0001F4E5 Extracting features from dataset...")
X, y = [], []
class_names = sorted(os.listdir(image_dir))
label_map = {name: idx for idx, name in enumerate(class_names)}

for class_name in tqdm(class_names):
    class_path = os.path.join(image_dir, class_name)
    for fname in os.listdir(class_path):
        fpath = os.path.join(class_path, fname)
        try:
            img = cv2.imread(fpath)
            if img is None:
                continue
            features = extract_features(img)
            if features.shape[0] == 34:
                X.append(features)
                y.append(label_map[class_name])
        except Exception as e:
            print(f"⚠ Error with {fname}: {e}")

X = np.array(X)
y = np.array(y)

# --- Train-Test Split, Scaling, SMOTE ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

# --- Train Ensemble Model ---
print("\n🚀 Training Ensemble (RF + XGB + LR)...")
clf1 = LogisticRegression(max_iter=1000)
clf2 = RandomForestClassifier(n_estimators=200)
clf3 = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=200)

ensemble = VotingClassifier(estimators=[
    ('lr', clf1),
    ('rf', clf2),
    ('xgb', clf3)
], voting='soft')

ensemble.fit(X_train_scaled, y_train)

# --- Evaluation ---
y_pred = ensemble.predict(X_test_scaled)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Boxed Format)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# --- ROC Curve (Multi-Class One-vs-Rest) ---
print("\n📈 Generating ROC Curve...")

# Binarize the labels
y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
y_score = ensemble.predict_proba(X_test_scaled)

# Compute ROC and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = len(class_names)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve (One-vs-Rest)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
print("✅ ROC Curve saved as 'roc_curve.png'")

# --- Save Trained Components ---
with open("fly_model.pkl", "wb") as f:
    pickle.dump(ensemble, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("\n✅ Model, scaler, and label map saved successfully!")
print("📊 Plot saved: confusion_matrix.png")
