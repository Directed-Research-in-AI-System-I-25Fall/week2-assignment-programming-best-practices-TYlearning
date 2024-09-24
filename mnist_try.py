
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
dataset = load_dataset("mnist")
total=0
correct=0
for i in range(1000):
    image=dataset["test"]["image"][i]
    rgb_image = image.convert("RGB")  # 将图像从灰度转换为 RGB
    inputs = processor(rgb_image, return_tensors="pt")
    total=total+1
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    if predicted_label==dataset["test"]["label"][i]:
        correct=correct+1
    print(f"Predicted: {predicted_label}, Actual: {dataset['test']['label'][i]}")
accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')
