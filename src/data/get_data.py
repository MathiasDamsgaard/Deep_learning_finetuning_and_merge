import kagglehub # type: ignore
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("seryouxblaster764/fgvc-aircraft")

# Move the downloaded dataset to the current directory
# Note, this is only if the data is made with the makefile target - remember to set this manually if otherwise
shutil.move(path, "/data") 
print("Dataset files moved to:", os.path.dirname(os.path.dirname(os.getcwd())) + "/data")