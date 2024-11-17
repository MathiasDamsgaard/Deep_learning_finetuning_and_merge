import os
import pandas as pd
import shutil
from tqdm import tqdm

DATA_DIR = os.path.join(os.getcwd(), "data", "resized_images")
CSV_DIR = os.path.join(os.getcwd(), 'data', 'seryouxblaster764/fgvc-aircraft/versions/2')

# Move resized images into train, val and test folders based on the images contained in the respective csv files
train_csv = pd.read_csv(os.path.join(CSV_DIR, 'train.csv'))
val_csv = pd.read_csv(os.path.join(CSV_DIR, 'val.csv'))
test_csv = pd.read_csv(os.path.join(CSV_DIR, 'test.csv'))

# Move images to respective folders
dst = os.path.join(DATA_DIR, 'train')
os.makedirs(dst, exist_ok=True)
for _, row in tqdm(train_csv.iterrows()):
    image = os.path.join(DATA_DIR, row['filename'])
    shutil.move(image, dst)

dst = os.path.join(DATA_DIR, 'val')
os.makedirs(dst, exist_ok=True)
for _, row in tqdm(val_csv.iterrows()):
    image = os.path.join(DATA_DIR, row['filename'])
    shutil.move(image, dst)

dst = os.path.join(DATA_DIR, 'test')
os.makedirs(dst, exist_ok=True)
for _, row in tqdm(test_csv.iterrows()):
    image = os.path.join(DATA_DIR, row['filename'])
    shutil.move(image, dst)