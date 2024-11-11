from peft import LoraConfig
from transformers import ViTImageProcessor, ViTModel, ViTForImageClassification
from PIL import Image
import requests
from matplotlib import pyplot as plt
import torch
import os

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(
    os.getcwd() + os.sep + "data" + os.sep + "resized_images" + os.sep + "2260088.jpg"
)


processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_states = outputs.logits
predicted_class_idx = torch.argmax(last_hidden_states).item()
print(model.config.id2label[predicted_class_idx])

print("Done!")
