import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import cv2
import joblib

# Set style
sns.set_style('darkgrid')

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (64, 64))  # Smaller size for ML models
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(gray.flatten())
                labels.append(1 if 'yes' in folder else 0)
    return images, labels

def main():
    # Load dataset
    data_dir = 'data'
    yes_dir = os.path.join(data_dir, 'yes')
    no_dir = os.path.join(data_dir, 'no')

    print("Loading images...")
    yes_images, yes_labels = load_images_from_folder(yes_dir)
    no_images, no_labels = load_images_from_folder(no_dir)

    X = np.array(yes_images + no_images)
    y = np.array(yes_labels + no_labels)

    print(f'Total images: {len(X)}')
    print(f'Tumor images: {np.sum(y)}')
    print(f'No tumor images: {len(y) - np.sum(y)}')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f'Train set: {X_train.shape[0]} samples')
    print(f'Test set: {X_test.shape[0]} samples')

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    print("Training SVM...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate SVM
    print("SVM Results:")
    y_pred_svm = svm_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_svm))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

    # Evaluate Random Forest
    print("Random Forest Results:")
    y_pred_rf = rf_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_rf))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

    # Save models
    joblib.dump(svm_model, 'models/svm_model.pkl')
    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Models saved!")

    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cm_svm = confusion_matrix(y_test, y_pred_svm)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('SVM Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_title('Random Forest Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('notebooks/ml_confusion_matrices.png')
    plt.show()

if __name__ == "__main__":
    main()
