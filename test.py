import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/monet2photo/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--model', type=str, default='checkpoints/style_monet.pth', help='model checkpoint file')
opt = parser.parse_args()
print(opt)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)

###### Definition of variables ######
# Networks
netG = Generator(opt.input_nc, opt.output_nc)

# if opt.cuda:
#     netG_A2B.cuda()
#     netG_B2A.cuda()

# Load state dicts
netG.load_state_dict(torch.load("checkpoints/style_monet.pth"), map_location=device)

# Set model's test mode
netG.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
model_name = "style_monet"

if not os.path.exists('checkpoints/' + model_name):
    os.makedirs('checkpoints/' + model_name)

for i, batch in enumerate(dataloader):
    # Set model input
    real = Variable(input.copy_(batch['A']))

    # Generate output
    fake = 0.5*(netG(real).data + 1.0)

    # Save image files
    save_image(fake, 'result/%04d.png' % (i+1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
