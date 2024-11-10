import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("seryouxblaster764/fgvc-aircraft")
print("Path to dataset files:", path)

USER = "Damsgaard" # Change this to your username
download_dir = f"C:/Users/{USER}/.cache/kagglehub/datasets/seryouxblaster764"
current_dir = os.getcwd() + "/data"

# Move the downloaded dataset to the current directory
shutil.move(download_dir, current_dir)
print("Dataset files moved to:", current_dir)