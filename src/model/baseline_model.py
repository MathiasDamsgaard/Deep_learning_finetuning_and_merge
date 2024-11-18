from peft import LoraConfig
from transformers import ViTImageProcessor, ViTModel, ViTForImageClassification
from PIL import Image
import requests
from matplotlib import pyplot as plt
import torch
import os
import pandas as pd
# from datasets import Dataset
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from src.config.config import *
import torch.nn as nn

processor = ViTImageProcessor.from_pretrained(MODEL)

# Show running device
print("Device:", DEVICE)
model = ViTForImageClassification.from_pretrained(MODEL).to(DEVICE)

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, processor, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        inputs = self.processor(images=image, return_tensors="pt")
        # convert inputs and label to dict object
        item = {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'label': label
        }
        return item

def load_dataset(csv_file: str = None, root_dir: str = None):
    """
    Load the dataset from the csv file and root directory.
    """
    dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, processor=processor)

    return dataset


def load_model(c: bool = False, path: str = None):
    """
    Load trained model if continiue training, else load the model from Huggingface.
    """
    if c:
        model = ViTForImageClassification.from_pretrained(path).to(DEVICE)
    else:
        model = ViTForImageClassification.from_pretrained(MODEL).to(DEVICE)
    
    # Change id2label to match the new number of classes
    train_df = pd.read_csv(TRAIN_CSV)

    # Extract unique labels
    unique_labels = train_df['Labels'].unique()

    # Create mappings
    id2label = {int(idx): str(label) for idx, label in enumerate(unique_labels)}
    label2id = {str(label): int(idx) for idx, label in enumerate(unique_labels)}

    # Update model classifier layer and mapping functions
    model.classifier = nn.Linear(model.config.hidden_size, len(id2label)).to(DEVICE)
    model.config.id2label = id2label
    model.config.label2id = label2id

    # Freeze base layers
    for param in model.base_model.parameters():
        param.requires_grad = False
    return model

def train_model(model, epochs: int):
    """
    Train the model for a number of epochs.
    """
    # Load the dataset
    dataset = load_dataset(csv_file=TRAIN_CSV, root_dir=TRAIN_DIR)

    # Define the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for batch in tqdm(dataset):
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
                 
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataset)}")

    return model

def save_model(model, path):
    """
    Save the model to a specified path.
    """
    model.save_pretrained(path)

def model_infer(model):
    """
    Infer the model on a single image.
    """
    preds = []
    test_df = pd.read_csv(TEST_CSV)

    # Loop through all images in test directory and predict the class. Save the prediction into the csv file.
    for i, row in tqdm(test_df.iterrows()):
        img_path = os.path.join(TEST_DIR, row['filename'])
        image = Image.open(img_path)
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        last_hidden_states = outputs.logits
        predicted_class_idx = torch.argmax(last_hidden_states).item()
        predicted_class = model.config.id2label[predicted_class_idx]
        preds.append(predicted_class)
    
    test_df['BM'] = preds
    test_df.to_csv(TEST_CSV, index=False)
    return preds

def eval_predictions():
    """
    Evaluate the predictions.
    """
    # Load the test csv file
    test_df = pd.read_csv(TEST_CSV)

    # Calculate the accuracy
    correct = 0
    for _, row in test_df.iterrows():
        if row['Labels'] == row['BM']:
            correct += 1

    accuracy = correct / len(test_df)
    return accuracy

if __name__ == "__main__":
    image = Image.open(
        os.getcwd() + os.sep + "data" + os.sep + "resized_images" + os.sep + "2260088.jpg"
    )
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    outputs = model(**inputs)
    last_hidden_states = outputs.logits
    predicted_class_idx = torch.argmax(last_hidden_states).item()
    print(model.config.id2label[predicted_class_idx])

    print("Done!")

    # Try to train the model
    model = train_model(epochs=1)
    print("Model trained successfully!")

