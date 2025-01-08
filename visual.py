import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path):
    return Image.open(image_path)

def visualize_results(demo_img_dir, results_dir):
    # 1. 获取图像与模型名字
    demo_images = sorted([
        fn for fn in os.listdir(demo_img_dir)
        if fn.lower().endswith(('.png','.jpg','.jpeg','.bmp','.gif'))
    ])
    model_names = sorted(os.listdir(results_dir))
    
    num_images = len(demo_images)
    num_models = len(model_names)
    
    # 2. 创建子图：每张原图 1 行，(模型数 + 1) 列
    #    调大 figsize 保证有足够的显示空间，可根据需要微调
    fig, axes = plt.subplots(
        nrows=num_images,
        ncols=num_models + 1,
        figsize=(4*(num_models + 1), 3*num_images)
    )

    # 如只有一张图或一个模型，axes 就不再是二维列表，要特殊处理
    # 可以用 np.atleast_2d() 规整为二维数组，便于统一索引
    axes = np.atleast_2d(axes)

    # 3. 遍历每张原图
    for i, img_name in enumerate(demo_images):
        real_img_path = os.path.join(demo_img_dir, img_name)
        real_img = load_image(real_img_path)

        # 第一列：原图
        axes[i, 0].imshow(real_img)
        axes[i, 0].set_title('Photograph', fontsize=12)
        axes[i, 0].axis('off')

        # 4. 其余列：各模型生成结果
        for j, model_name in enumerate(model_names):
            # 结果图像路径，例如 "results/Monet/xxx_fake.png"
            # 注意这里 img_name[:-4] 可能需要再判断后缀长度，
            # 或者可直接把拼出来的路径做个简单检查
            base, ext = os.path.splitext(img_name)
            result_img_path = os.path.join(results_dir, model_name, f'{base}_fake.png')
            
            # 若不存在对应结果，可自行选择跳过或抛异常
            if not os.path.exists(result_img_path):
                print(f'Warning: {result_img_path} not found, skipped.')
                axes[i, j+1].axis('off')
                continue
            
            result_img = load_image(result_img_path)
            axes[i, j+1].imshow(result_img)
            
            # 你可以在这里对 model_name 做一层映射，比如
            # if model_name.lower() == 'monet': show_name = 'Monet'
            # else: show_name = model_name
            # 这里只是直接显示文件夹名
            axes[i, j+1].set_title(model_name, fontsize=12)
            axes[i, j+1].axis('off')

    plt.tight_layout()
    plt.savefig('results.png', dpi=600)

# 使用示例
if __name__ == '__main__':
    demo_img_dir = 'demo_img'
    results_dir = 'results'
    visualize_results(demo_img_dir, results_dir)