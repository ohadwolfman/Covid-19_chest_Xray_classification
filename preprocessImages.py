import torch
import numpy as np
import cv2
import os

# Function to preprocess image
def preprocess_image(image_path, target_size=(300, 300)):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray_img, target_size)
    img_flattened = img_resized.flatten()
    img_normalized = img_flattened.astype("float32") / 255.0
    return img_normalized

def preprocess_data():
    # Load and preprocess train and test data
    train_dir = 'train'
    test_dir = 'test'

    train_filepaths = []
    train_labels = []
    for label in os.listdir(train_dir):
        label_dir = os.path.join(train_dir, label)
        for filename in os.listdir(label_dir):
            filepath = os.path.join(label_dir, filename)
            train_filepaths.append(filepath)
            train_labels.append(label)

    test_filepaths = []
    test_labels = []
    for label in os.listdir(test_dir):
        label_dir = os.path.join(test_dir, label)
        for filename in os.listdir(label_dir):
            filepath = os.path.join(label_dir, filename)
            test_filepaths.append(filepath)
            test_labels.append(label)

    target_size = (300, 300)
    x_train = np.array([preprocess_image(filepath, target_size) for filepath in train_filepaths])
    y_train = np.array(train_labels)
    x_test = np.array([preprocess_image(filepath, target_size) for filepath in test_filepaths])
    y_test = np.array(test_labels)

    # Convert labels to numerical categories
    label_to_index = {label: i for i, label in enumerate(np.unique(y_train))}
    y_train = np.array([label_to_index[label] for label in y_train])
    y_test = np.array([label_to_index[label] for label in y_test])

    # Convert NumPy arrays to PyTorch tensors
    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train).long()
    x_test_tensor = torch.from_numpy(x_test)
    y_test_tensor = torch.from_numpy(y_test).long()

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor
