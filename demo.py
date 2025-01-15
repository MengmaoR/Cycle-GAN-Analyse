import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import SimpleImageDataset

def demo(model_name, dataloader, device, Tensor):
    model_path = f"checkpoints/{model_name}/netG_B2A.pth"

    # 尝试加载模型
    try:
        model = torch.load(model_path, map_location=device)
        netG = Generator(3, 3).to(device)
        netG.load_state_dict(model, strict=False)
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

    netG.eval()

    save_path = f"results/{model_name}"

    if not os.path.exists('results/' + model_name):
        os.makedirs('results/' + model_name)

    for i, batch in enumerate(dataloader):
        real_A = Variable(batch.type(Tensor))
        fake = 0.5*(netG(real_A)[0].data + 1.0)

        save_image(fake, f'{save_path}/%04d_fake.png' % (i+1))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

    sys.stdout.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="typical", help="Name of the model to use")
    parser.add_argument("--dataroot", type=str, default="./demo_img", help="Path to the images to be converted")
    opt = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    transforms_ = [ transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    
    img_path = opt.dataroot
    dataloader = DataLoader(
        SimpleImageDataset(img_path, transforms_=transforms_),
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    demo(opt.model_name, dataloader, device, Tensor)

if __name__ == "__main__":
    main()