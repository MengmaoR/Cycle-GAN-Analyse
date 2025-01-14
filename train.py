import argparse
import itertools
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from models import PatchGAN16, PatchGAN70, PatchGAN142
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/monet2photo/', help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--use_pretrained', action='store_true', help='use pretrained model')
    parser.add_argument('--style', type=str, default='monet', help='style name')
    parser.add_argument('--patch', type=int, default=70, help='patch size')
    opt = parser.parse_args()
    print(opt)

    model_path = f"checkpoints/attention"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc).to(device)
    netG_B2A = Generator(opt.output_nc, opt.input_nc).to(device)
    
    if opt.patch == 16:
        netD_A = PatchGAN16(opt.input_nc).to(device)
        netD_B = PatchGAN16(opt.output_nc).to(device)
    elif opt.patch == 70:
        netD_A = PatchGAN70(opt.input_nc).to(device)
        netD_B = PatchGAN70(opt.output_nc).to(device)
    elif opt.patch == 142:
        netD_A = PatchGAN142(opt.input_nc).to(device)
        netD_B = PatchGAN142(opt.output_nc).to(device)
    else:
        raise ValueError(f"Invalid patch size: {opt.patch}")

    # 如果不使用预训练，则对网络进行随机初始化
    if not opt.use_pretrained:
        netG_A2B.apply(weights_init_normal)
        netG_B2A.apply(weights_init_normal)
        netD_A.apply(weights_init_normal)
        netD_B.apply(weights_init_normal)
    else:
        print(f"Using pretrained model, loading from {model_path}")
        # 使用预训练模型进行二次训练
        try:
            if os.path.exists(f'{model_path}/netG_B.pth'):
                ga = torch.load(f'{model_path}/netG_B.pth', map_location=device)
                netG_A2B.load_state_dict(ga)
            else:
                netG_A2B.apply(weights_init_normal)
            
            if os.path.exists(f'{model_path}/netD_A.pth'):
                da = torch.load(f'{model_path}/netD_A.pth', map_location=device)
                netD_A.load_state_dict(da)
            else:
                netD_A.apply(weights_init_normal)
            
            if os.path.exists(f'{model_path}/netG_A.pth'):
                gb = torch.load(f'{model_path}/netG_A.pth', map_location=device)
                netG_B2A.load_state_dict(gb)
            else:
                netG_B2A.apply(weights_init_normal)

            if os.path.exists(f'{model_path}/netD_B.pth'):
                db = torch.load(f'{model_path}/netD_B.pth', map_location=device)
                netD_B.load_state_dict(db)
            else:
                netD_B.apply(weights_init_normal)


            print(f"Pretrained Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize, 1, 30, 30).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize, 1, 30, 30).fill_(0.0), requires_grad=False)
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                    transforms.RandomCrop(opt.size), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
    
    # Loss plot
    logger = Logger(opt.n_epochs, len(dataloader))
    ###################################

    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            real_A_img = (real_A[0].cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2.0 * 255.0
            real_A_img = Image.fromarray(real_A_img.astype('uint8'))
            real_A_img.save(f'output/real_A_epoch{epoch}_batch{i}.png')
            real_B_img = (real_B[0].cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2.0 * 255.0
            real_B_img = Image.fromarray(real_B_img.astype('uint8'))
            real_B_img.save(f'output/real_B_epoch{epoch}_batch{i}.png')

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B, _ = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5.0
            # same_B_img = (same_B[0].cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2.0 * 255.0
            # same_B_img = Image.fromarray(same_B_img.astype('uint8'))
            # same_B_img.save(f'output/same_B_epoch{epoch}_batch{i}.png')
            # G_B2A(A) should equal A if real A is fed

            same_A, _ = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5.0
            # same_A_img = (same_A[0].cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2.0 * 255.0
            # same_A_img = Image.fromarray(same_A_img.astype('uint8'))
            # same_A_img.save(f'output/same_A_epoch{epoch}_batch{i}.png')

            # GAN loss
            fake_B, _ = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            print(f"expand: {target_real.shape}")
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
            print(f"loss_GAN_A2B: {loss_GAN_A2B}")

            target_real_this = torch.ones_like(pred_fake)
            loss_GAN_A2B_this = criterion_GAN(pred_fake, target_real_this)
            print(f"loss_GAN_A2B_this: {loss_GAN_A2B_this}")

            fake_B_img = (fake_B[0].cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2.0 * 255.0
            fake_B_img = Image.fromarray(fake_B_img.astype('uint8'))
            fake_B_img.save(f'output/fake_B_epoch{epoch}_batch{i}.png')

            fake_A, _ = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real.expand_as(pred_fake))

            fake_A_img = (fake_A[0].cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2.0 * 255.0
            fake_A_img = Image.fromarray(fake_A_img.astype('uint8'))
            fake_A_img.save(f'output/fake_A_epoch{epoch}_batch{i}.png')

            # Cycle loss
            recovered_A, _ = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B, _ = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            
            optimizer_G.step()
            ###################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################

            # Progress report (http://localhost:8097)
            logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                        'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
                        images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        if os.path.exists(f'output/netG_A2B_e{epoch-1}.pth'):
            os.remove(f'output/netG_A2B_e{epoch-1}.pth')
        if os.path.exists(f'output/netG_B2A_e{epoch-1}.pth'):
            os.remove(f'output/netG_B2A_e{epoch-1}.pth')
        if os.path.exists(f'output/netD_A_e{epoch-1}_p{opt.patch}.pth'):
            os.remove(f'output/netD_A_e{epoch-1}_p{opt.patch}.pth')
        if os.path.exists(f'output/netD_B_e{epoch-1}_p{opt.patch}.pth'):
            os.remove(f'output/netD_B_e{epoch-1}_p{opt.patch}.pth')

        torch.save(netG_A2B.state_dict(), f'output/netG_A2B_e{epoch}.pth')
        torch.save(netG_B2A.state_dict(), f'output/netG_B2A_e{epoch}.pth')
        torch.save(netD_A.state_dict(), 'output/netD_A_e{epoch}_p{opt.patch}.pth')
        torch.save(netD_B.state_dict(), 'output/netD_B_e{epoch}_p{opt.patch}.pth')
    ###################################

if __name__ == "__main__":
    main()