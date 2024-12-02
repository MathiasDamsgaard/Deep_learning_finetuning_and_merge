import os
import pandas as pd
import shutil
from tqdm import tqdm

DATA_DIR = os.path.join(os.getcwd(), "data")
DST_DIR = os.path.join(DATA_DIR, "resized_images")
CSV_DIR = os.path.join(os.getcwd(), 'data', 'seryouxblaster764/fgvc-aircraft/versions/2')

# Move resized images into train, val and test folders based on the images contained in the respective csv files
train_csv = pd.read_csv(os.path.join(CSV_DIR, 'train.csv'))
val_csv = pd.read_csv(os.path.join(CSV_DIR, 'val.csv'))
test_csv = pd.read_csv(os.path.join(CSV_DIR, 'test.csv'))

# Copy the csv files to the data directory
shutil.copy(os.path.join(CSV_DIR, 'train.csv'), DATA_DIR)
shutil.copy(os.path.join(CSV_DIR, 'val.csv'), DATA_DIR)
shutil.copy(os.path.join(CSV_DIR, 'test.csv'), DATA_DIR)

# Move images to respective folders
os.makedirs(os.path.join(DST_DIR, 'train'), exist_ok=True)
for _, row in tqdm(train_csv.iterrows()):
    image = os.path.join(DST_DIR, row['filename'])
    dst = os.path.join(DST_DIR, 'train', row['filename'])
    shutil.move(image, dst)

os.makedirs(os.path.join(DST_DIR, 'val'), exist_ok=True)
for _, row in tqdm(val_csv.iterrows()):
    image = os.path.join(DST_DIR, row['filename'])
    dst = os.path.join(DST_DIR, 'val', row['filename'])
    shutil.move(image, dst)

os.makedirs(os.path.join(DST_DIR, 'test'), exist_ok=True)
for _, row in tqdm(test_csv.iterrows()):
    image = os.path.join(DST_DIR, row['filename'])
    dst = os.path.join(DST_DIR, 'test', row['filename'])
    shutil.move(image, dst)