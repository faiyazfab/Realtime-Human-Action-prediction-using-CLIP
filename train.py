from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def handle_missing_images(csv_file, image_folder):
    """
    Handles missing image files in a dataset.

    Args:
        csv_file (str): Path to the CSV file.
        image_folder (str): Path to the folder containing images.
    """

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return

    try:
        image_files = os.listdir(image_folder)
    except FileNotFoundError:
        print(f"Error: Image folder '{image_folder}' not found.")
        return

    # Extract filenames from the CSV
    csv_filenames = df['filename'].tolist()

    # Find missing files
    missing_files = []
    for filename in csv_filenames:
        if filename not in image_files:
            missing_files.append(filename)

    if missing_files:
        print(f"Found {len(missing_files)} missing image files:")
        for filename in missing_files:
            print(filename)

        # Option 1: Remove rows with missing files
        df_cleaned = df[~df['filename'].isin(missing_files)].copy()
        print(f"Removed {len(missing_files)} rows from the DataFrame.")

        # Option 2: Log missing files to a file
        with open("missing_images.log", "w") as log_file:
            for filename in missing_files:
                log_file.write(filename + "\n")
        print("Missing files logged to 'missing_images.log'.")

        return df_cleaned
    else:
        print("No missing image files found.")
        return df
    

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_file, preprocess, limit=500):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            annotation_file (str): Path to the CSV file with image-caption mappings.
            preprocess (callable): Preprocessing function for images.
            limit (int): Maximum number of entries to include in the dataset.
        """
        self.image_dir = image_dir
        self.annotations = annotation_file.head(limit)  # Limit dataset to first 'limit' rows
        self.preprocess = preprocess

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the filename and label from the CSV file
        img_info = self.annotations.iloc[idx]
        img_filename = img_info['filename']
        caption = img_info['label']

        # Construct the full path to the image file
        img_path = os.path.join(self.image_dir, img_filename)

        # Check if the file exists (optional safety check)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # Load and preprocess the image
        image = self.preprocess(Image.open(img_path).convert("RGB"))

        # Tokenize the caption using CLIP's tokenizer
        text = clip.tokenize([caption])[0]

        return image, text
    
image_dir = "d:/Projects/Patient_montioring_system/Train/train"
annotation_file = "d:/Projects/Patient_montioring_system/Train/Training_set.csv" 

cleaned_df = handle_missing_images(annotation_file, image_dir)

dataset = CustomDataset(image_dir=image_dir, annotation_file=cleaned_df, preprocess=preprocess)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, texts in dataloader:
        images, texts = images.to(device), texts.to(device)

        # Forward pass
        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        
        # Compute loss
        loss = (loss_fn(logits_per_image, ground_truth) + loss_fn(logits_per_text, ground_truth)) / 2

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), "fine_tuned_clip_vit_b32.pth")