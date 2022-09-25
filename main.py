import os 
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN


def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)
    for i, image in enumerate(os.listdir("inputs")):
        image = Image.open(f"inputs/{image}").convert('RGB')
        sr_image = model.predict(image)
        sr_image.save(f'results/{i}.png')


if __name__ == '__main__':
    main()