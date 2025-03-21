# Realtime Human Action Recognition using CLIP

## Description
This project implements real-time human action recognition using OpenAI's CLIP model fine-tuned with a custom human action recognition dataset. The model processes webcam input to predict and display human actions based on predefined prompts.

## Features
- Fine-tunes CLIP (ViT-B/32) on a custom dataset with image-action mappings.
- Utilizes PyTorch for training and inference.
- Real-time action classification using a webcam.
- Displays predictions as text overlays on the video feed.

## Dataset
The training dataset consists of images labeled with corresponding human actions. The dataset is provided as a CSV file containing image filenames and their respective labels.

## Installation
### Requirements
Ensure you have the following dependencies installed:
```sh
pip install -r requirements.txt
```

## Usage
### Train the Model
Modify the dataset path and run the training script:
```sh
python train.py
```

### Run Real-time Action Recognition
Execute the script to start webcam-based inference:
```sh
python action_recognition.py
```
Press `q` to exit the real-time recognition.

## Training Process
1. Loads the CLIP model and dataset.
2. Fine-tunes the model using image-text embeddings.
3. Saves the trained model as `fine_tuned_clip_vit_b32.pth`.

## Real-time Inference
1. Captures video frames from the webcam.
2. Encodes frames using CLIP’s image processing pipeline.
3. Computes similarity scores between image features and predefined action prompts.
4. Displays the highest probability action on the video feed.

## Example Actions
- `using_laptop`
- `hugging`
- `drinking`
- `texting`

## Model Checkpoint
The fine-tuned model is saved as `fine_tuned_clip_vit_b32.pth` and can be reloaded for inference.

## Future Improvements
- Expand dataset with diverse actions.
- Optimize real-time inference speed.
- Implement multi-person action recognition.

## License
This project is licensed under the MIT License.

## Acknowledgements
- OpenAI for CLIP
- PyTorch for deep learning framework
- OpenCV for real-time video processing

