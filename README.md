# Real-ESRGAN
PyTorch implementation of a Real-ESRGAN model trained on custom dataset. This model shows better results on faces compared to the original version. It is also easier to integrate this model into your projects.

You can try it in [google colab](https://colab.research.google.com/drive/1yO6deHTscL7FBcB6_SRzbxRr1nVtuZYE?usp=sharing)

- Paper: [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
- [Official github](https://github.com/xinntao/Real-ESRGAN)

### Installation

---

1. Clone repo

   ```bash
   git clone https://https://github.com/sberbank-ai/Real-ESRGAN
   cd Real-ESRGAN
   ```

2. Install requirements

   ```bash
   pip install -r requirements.txt
   ```

3. Download [pretrained weights](https://drive.google.com/drive/folders/16PlVKhTNkSyWFx52RPb2hXPIQveNGbxS) and put them into `weights/` folder

### Usage

---

Basic example:

```python
import torch
from PIL import Image
import numpy as np
from realesrgan import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth')

path_to_image = 'inputs/lr_image.png'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('results/sr_image.png')
```

