from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import time


def batch_inference():
    # 1️⃣ Load your fine-tuned model
    model_dir = "./asl-model-v3"
    image_processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(model_dir)

    # 2️⃣ Setup device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3️⃣ Load multiple images (list of paths)
    image_paths = [
        "sample.png",
        "sample.png",
    ]  # 🔹 add as many as you want

    images = [Image.open(p).convert("RGB") for p in image_paths]

    # 4️⃣ Preprocess the batch
    inputs = image_processor(images, return_tensors="pt").to(device)

    # 5️⃣ Inference (FP16 if GPU available)
    with torch.no_grad():
        start = time.time()

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):  # ✅ mixed precision
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

        end = time.time()
        elapsed_ms = (end - start) * 1000  # ms

        logits = outputs.logits
        probs = logits.softmax(dim=-1)

    # 6️⃣ Get predictions for each image
    for i, path in enumerate(image_paths):
        predicted_class_idx = logits[i].argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]
        confidence = probs[i, predicted_class_idx].item()

        print(f"[{path}] → {predicted_label} (confidence: {confidence:.2f})")

    print(f"\nBatch size: {len(image_paths)}")
    print(f"Total inference time: {elapsed_ms:.2f} ms on {device.type.upper()}")
    print(f"Avg per image: {elapsed_ms/len(image_paths):.2f} ms")
