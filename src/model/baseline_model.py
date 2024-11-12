from peft import LoraConfig
from transformers import ViTImageProcessor, ViTModel, ViTForImageClassification
from PIL import Image
import requests
from matplotlib import pyplot as plt
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from src.config.config import MODEL, IN_DIM, DEVICE

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
        return inputs['pixel_values'].squeeze(), label

def load_dataset(batch_size=32):
    csv_file = os.path.join(os.getcwd(), 'data', 'data.csv')
    root_dir = os.path.join(os.getcwd(), 'data', 'resized_images')
    dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, processor=processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def load_model(c: bool = False, path: str = None):
    """
    Load trained model if continiue training, else load the model from Huggingface.
    """
    if c:
        model = ViTForImageClassification.from_pretrained(path).to(DEVICE)
    else:
        model = ViTForImageClassification.from_pretrained(MODEL).to(DEVICE)
    return model

def train_model(model, epochs: int):
    """
    Train the model for a number of epochs.
    """
    # Load the dataset
    dataset = load_dataset()

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

