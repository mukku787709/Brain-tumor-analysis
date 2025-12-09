import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

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
                img = cv2.resize(img, (128, 128))
                images.append(img)
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

    # Normalize
    X = X / 255.0

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f'Train set: {X_train.shape[0]} samples')
    print(f'Test set: {X_test.shape[0]} samples')

    # Build CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Training CNN...")
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    print("CNN Results:")
    print(classification_report(y_test, y_pred))

    # Save model
    model.save('models/cnn_model.h5')
    print("Model saved!")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('notebooks/dl_training_history.png')
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('CNN Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('notebooks/dl_confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    main()
