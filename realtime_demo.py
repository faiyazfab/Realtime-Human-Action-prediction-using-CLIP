import cv2
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

model.load_state_dict(torch.load("fine_tuned_clip_vit_b32.pth"))
model.eval()

text_prompts = ["using_laptop", "hugging","drinking","texting"]
text_tokens = clip.tokenize(text_prompts).to(device)

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for CLIP model input
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_input = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode image and text features
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

        # Compute similarity scores and probabilities
        logits_per_image = (image_features @ text_features.T).softmax(dim=-1)
    
    probs = logits_per_image.cpu().numpy()[0]

    # Display predictions on the video feed
    for i, prompt in enumerate(text_prompts):
        cv2.putText(frame, f"{prompt}: {probs[i]:.2f}", (10, 30 + i * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("CLIP Webcam", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
