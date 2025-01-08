import os
from PIL import Image, ImageEnhance

# Define input and output directories
input_dir = './my_img/'
output_dir = './demo_img/'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Enhance for all images in the input directory
i = 0
for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.JPG', '.jpg', '.jpeg', '.bmp', '.gif')):
        # Open an image file
        with Image.open(os.path.join(input_dir, filename)) as img:
            # Enhance brightness
            enhancer = ImageEnhance.Brightness(img)
            # Check the brightness of the image
            grayscale_img = img.convert('L')
            brightness = grayscale_img.getextrema()[1]
            
            # Define a brightness threshold
            brightness_lower_bound = 200
            brightness_upper_bound = 300
            
            # Enhance brightness if below the threshold
            while brightness < brightness_lower_bound:
                print(f"Enhancing brightness of {filename}")
                img = enhancer.enhance(1.2)  # Increase brightness by a factor of 1.2
                brightness = img.convert('L').getextrema()[1]
            
            while brightness > brightness_upper_bound:
                print(f"Enhancing brightness of {filename}")
                img = enhancer.enhance(0.8)

            # Resize image to have the longer side equal to 512 pixels
            size = 1080
            width, height = img.size
            if width > height:
                new_width = size
                new_height = int((height / width) * size)
            else:
                new_height = size
                new_width = int((width / height) * size)
            img = img.resize((new_width, new_height), Image.LANCZOS)

            # Save the enhanced image to the output directory
            img.save(os.path.join(output_dir, '%04d.png' % (i+1)))
            i += 1

print("Contrast enhancement completed.")