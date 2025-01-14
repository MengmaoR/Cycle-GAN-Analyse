import os
from PIL import Image, ImageEnhance

# 输入和输出目录
input_dir = './my_img/'
output_dir = './demo_img/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

i = 0
for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.JPG', '.jpg', '.jpeg', '.bmp', '.gif')):
        with Image.open(os.path.join(input_dir, filename)) as img:
            # 增强亮度
            enhancer = ImageEnhance.Brightness(img)
            # 检查图像的亮度
            grayscale_img = img.convert('L')
            brightness = grayscale_img.getextrema()[1]
            
            # 定义亮度阈值
            brightness_lower_bound = 200
            brightness_upper_bound = 300
            
            # 如果亮度低于阈值，则增强亮度
            while brightness < brightness_lower_bound:
                print(f"Enhancing brightness of {filename}")
                img = enhancer.enhance(1.2)  # 将亮度增加1.2倍
                brightness = img.convert('L').getextrema()[1]
            
            # 如果亮度高于阈值，则降低亮度
            while brightness > brightness_upper_bound:
                print(f"Enhancing brightness of {filename}")
                img = enhancer.enhance(0.8)

            # 将图像调整为长边等于512像素
            size = 1080
            width, height = img.size
            if width > height:
                new_width = size
                new_height = int((height / width) * size)
            else:
                new_height = size
                new_width = int((width / height) * size)
            img = img.resize((new_width, new_height), Image.LANCZOS)

            img.save(os.path.join(output_dir, '%04d.png' % (i+1)))
            i += 1