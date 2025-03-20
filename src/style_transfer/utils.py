import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO

# Image Loader and Preprocessing
def ImageLoader(image_bytes, size=(520, 520)):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    loader = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return loader(image).unsqueeze(0)