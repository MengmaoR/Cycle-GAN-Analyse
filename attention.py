import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import models
from PIL import Image
from torchvision import transforms
import os

def resize_attention_to_image(attn_map, target_height, target_width):
    if isinstance(attn_map, np.ndarray):
        # 转成 tensor
        attn_map = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0)  # shape = (1,1,H',W')
    else:
        if attn_map.dim() == 2:
            attn_map = attn_map.unsqueeze(0).unsqueeze(0)
        elif attn_map.dim() == 3:
            attn_map = attn_map.unsqueeze(0)
    
    attn_resized = F.interpolate(attn_map, size=(target_height, target_width), mode='bilinear', align_corners=False)
    attn_resized = attn_resized.squeeze().cpu().numpy()  # (H, W)
    return attn_resized

def overlay_attention_on_image(img_np, attn_map, alpha=0.4, cmap='jet'):
    """
    将单通道的 attn_map 叠加到原图 img_np 上 (H,W,3).
    - img_np: (H,W,3), 0~1 或 0~255
    - attn_map: (H,W), 0~1
    - alpha: 叠加透明度
    - cmap: 使用的颜色映射, 如 'jet', 'viridis' 等.
    """
    if img_np.max() > 1.0:
        # 若图像是0~255, 先归一化到0~1
        img_np = img_np / 255.0
    
    # 用matplotlib的 colormap 转成 RGBA (H,W,4)
    colored_attn = plt.get_cmap(cmap)(attn_map)[:, :, :3]
    # alpha混合
    overlay = (1 - alpha) * img_np + alpha * colored_attn
    overlay = np.clip(overlay, 0, 1)
    return overlay

def visualize_attention_cycleGAN(netG_A2B, real_A):
    """
    1) 使用 netG_A2B 做推理, 得到 fake_B 和 attn_maps
    2) 将 attn_maps 与 real_A 做叠加可视化
    """
    device = next(netG_A2B.parameters()).device  # 当前模型所在device
    netG_A2B.eval()
    
    real_A = real_A.to(device)
    with torch.no_grad():
        fake_B, attn_maps = netG_A2B(real_A)  # 假设此时返回 attn_maps

    # 假设 batch_size=1, 取下标 0
    real_A = 0.5 * (real_A + 1.0)
    real_A_np = real_A[0].cpu().numpy()  # shape=(3,H,W)
    real_A_np = np.transpose(real_A_np, (1,2,0))  # => (H,W,3)

    # 画个大图, 同时显示 input, fake_B, 以及若干注意力叠加结果
    fig, axes = plt.subplots(1, 2 + 2*len(attn_maps), figsize=(4*(2+len(attn_maps)), 4))
    
    # 显示原图
    axes[0].imshow(real_A_np.astype('float32')/255. if real_A_np.max()>1 else real_A_np)
    axes[0].set_title("Input: real_A")
    axes[0].axis('off')
    
    # 显示生成的 fake_B
    fake_B = 0.5*(fake_B.data + 1.0)
    fake_B_np = fake_B[0].cpu().numpy()  # shape=(3,H,W)
    fake_B_np = np.transpose(fake_B_np, (1,2,0))
    axes[1].imshow(fake_B_np.astype('float32')/255. if fake_B_np.max()>1 else fake_B_np)
    axes[1].set_title("Output: fake_B")
    axes[1].axis('off')
    
    # 假设 attn_maps 是个 list, 每个元素形状可能不同.
    # 下面演示逐个可视化
    for i, attn_map in enumerate(attn_maps):
        # 例: 如果 attn_map 的形状是 (1, H', W'), 先 squeeze
        # 如果是 (N, N) 也要先 reshape or 选择某个 query
        # 这里只做最简单的假设, 即 attn_map = (1,H',W') for single batch
        attn_map_i = attn_map[0]  # => shape=(H',W') or (C,H',W')
        
        # 若是多通道, 你可能要先做平均/取最大再可视化
        if attn_map_i.dim() == 3:
            # eg: shape=(C,H',W') -> 做通道平均
            attn_map_i = attn_map_i.mean(dim=0)  # => (H',W')
        
        attn_map_i = attn_map_i.detach().cpu()
        
        # 缩放到跟 real_A_np 一样的size
        H, W = real_A_np.shape[:2]
        attn_map_resized = resize_attention_to_image(attn_map_i, H, W)
        axes[2 + 2*i].imshow(attn_map_resized, cmap='jet')
        axes[2 + 2*i].set_title(f"AttnMap")
        axes[2 + 2*i].axis('off')
        
        # 归一化到 [0,1]
        attn_min, attn_max = attn_map_resized.min(), attn_map_resized.max()
        if attn_max - attn_min < 1e-5:
            attn_map_norm = np.zeros_like(attn_map_resized)
        else:
            attn_map_norm = (attn_map_resized - attn_min) / (attn_max - attn_min)
        
        # 与输入图像叠加
        overlay_img = overlay_attention_on_image(real_A_np, attn_map_norm, alpha=0.4, cmap='jet')
        
        axes[2 + 2*i+1].imshow(overlay_img)
        axes[2 + 2*i+1].set_title(f"Overlay")
        axes[2 + 2*i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('attention_visualization.png')

def main():
    netG_A2B = models.Generator(3, 3).to('cpu')
    netG_A2B.load_state_dict(torch.load('checkpoints/attention/netG_B2A.pth', map_location=torch.device('cpu')))
    # 读取图片
    img_path = os.path.join('demo_img/0012.png')
    img = Image.open(img_path).convert('RGB')

    # 转换为 tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    real_A = transform(img).unsqueeze(0)  # 增加 batch 维度
    
    # 2. 可视化
    visualize_attention_cycleGAN(netG_A2B, real_A)

if __name__ == '__main__':
    main()