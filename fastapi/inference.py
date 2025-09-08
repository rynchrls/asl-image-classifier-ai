from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import time


def inference(image):

    # 1️⃣ Load your fine-tuned model
    model_dir = "./asl-model"
    image_processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(model_dir)

    # 2️⃣ Setup device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3️⃣ Load the image
    image_path = image  # make sure this exists
    image = Image.open(image_path).convert("RGB")

    # 4️⃣ Preprocess the image
    inputs = image_processor(image, return_tensors="pt").to(device)

    # 5️⃣ Inference (FP16 if GPU available)
    with torch.no_grad():
        start = time.time()

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):  # ✅ new API
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

        end = time.time()
        elapsed_ms = (end - start) * 1000  # ms

        logits = outputs.logits

    # 6️⃣ Get predicted class
    probs = logits.softmax(dim=-1)
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    confidence = probs[0, predicted_class_idx].item()

    print(f"Predicted class: {predicted_label} (confidence: {confidence:.2f})")
    print(f"Inference time: {elapsed_ms:.2f} ms on {device.type.upper()}")

    # 7️⃣ Show Top-5 predictions
    topk = torch.topk(probs, k=5)
    for i, (idx, score) in enumerate(zip(topk.indices[0], topk.values[0])):
        print(f"{i+1}: {model.config.id2label[idx.item()]} ({score.item():.2f})")

    return predicted_label
