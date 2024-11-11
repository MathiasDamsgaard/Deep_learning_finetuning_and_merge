import os
from PIL import Image, ImageOps
import pandas as pd
from tqdm import tqdm



# Define the source and destination directories
source_dir = os.getcwd() + '/data'
data_dir = os.path.join(source_dir, 'seryouxblaster764/fgvc-aircraft/versions/2/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images')
destination_dir = os.path.join(source_dir, 'resized_images')

csv_dir = os.path.join(source_dir, 'seryouxblaster764/fgvc-aircraft/versions/2')
csv_files = ['train.csv', 'val.csv', 'test.csv']
output_csv = os.path.join(source_dir, 'data.csv')


# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Define the new size
new_size = (224, 224)

# Loop through all files in the source directory
for filename in tqdm(os.listdir(data_dir)):
    if filename.endswith('.jpg'):
        # Open an image file
        with Image.open(os.path.join(data_dir, filename)) as img:
            # Remove the bottom 15 pixels
            img = img.crop((0, 0, img.width, img.height - 15))
            # preserve the aspect ratio so padding is needed
            width, height = img.size
            if width > height:
                new_height = int(new_size[0] * height / width)
                padding = (0, (new_size[1] - new_height) // 2, 0, (new_size[1] - new_height) // 2)
            else:
                new_width = int(new_size[1] * width / height)
                padding = ((new_size[0] - new_width) // 2, 0, (new_size[0] - new_width) // 2, 0)
            
            img = ImageOps.expand(img, padding, fill='white')

            # Resize the image
            resized_img = img.resize(new_size, Image.LANCZOS)
            # Save it to the destination directory
            resized_img.save(os.path.join(destination_dir, filename))

print(f"All images have been resized to {new_size}, and saved to {destination_dir}")


# Combine CSV files
combined_df = pd.concat([pd.read_csv(os.path.join(csv_dir, f)) for f in csv_files], ignore_index=True)
combined_df.reset_index(drop=True, inplace=True)  # Reset row indices

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(output_csv, index=False)
print(f"Combined CSV saved at: {output_csv}")
