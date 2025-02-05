 # Cycle-GAN-Analyse
![demo](./imgs/demo.png)

## 项目简介
本项目基于 Cycle GAN，主要对其在图像风格迁移能力方面进行分析和改进。
Cycle GAN 是一种无监督学习模型，能够在不需要成对训练数据的情况下实现图像到图像的转换。它的核心思想是通过两个生成器和两个判别器，将两个不同领域的图像进行转换。其中，生成器 G 将一个领域的图像转换为另一个领域的图像，生成器 F 则将另一个领域的图像转换回第一个领域。判别器 D 用于判断生成的图像是否真实，判别器 E 用于判断转换后的图像是否与原图像一致。通过这种方式，Cycle GAN 可以实现图像的风格迁移。
Cycle GAN 原项目地址：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix?tab=readme-ov-file

## 实验环境准备
首先，请克隆本项目到本地：
```bash
git clone https://github.com/MengmaoR/Cycle-GAN-Analyse
cd Cycle-GAN
```

### 环境依赖配置
请使用以下命令安装所需的依赖：
```bash
pip install -r requirements.txt
```

### 数据集下载
本项目使用 Cycle GAN 官方提供的数据集进行，你可以通过如下命令下载并解压数据集：
```bash
bash ./download_dataset.sh dataset_name
```
可选的`dataset_name` 有 `apple2orange`, `summer2winter_yosemite`, `horse2zebra`, `monet2photo`, `cezanne2photo`, `ukiyoe2photo`, `vangogh2photo`等 。

或者，你也可以直接访问 Cycle GAN 官方的数据集下载链接手动下载：http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/

### 预训练模型下载
本项目支持使用预训练模型进行训练和测试，你可以通过如下命令下载 Cycle GAN 官方预训练模型：
```bash
bash ./download_model.sh model_name
```
可选的的 `model_name` 有 `apple2orange`,  `summer2winter_yosemite`, `monet2photo`, `style_monet`, `style_cezanne`, `style_ukiyoe`, `style_vangogh`等 。

同样的，你也可以直接访问 Cycle GAN 官方的预训练模型下载链接手动下载：http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/

## 实验方法

### 模型训练
你可以通过如下命令训练 Cycle GAN 模型：
```bash
python train.py --dataroot ./datasets/dataset_name
```
对于其余参数的，请参阅 `train.py` 文件。

训练过程中，模型会自动保存在 `./output/` 目录下。如果你希望实时查看训练过程，请使用 Visdom：
```bash
pip install visdom
python -m visdom
```
然后在浏览器中打开 `http://localhost:8097/` 即可查看训练过程。

训练结束后，完整的训练日志会被保存在 `training_log.txt` 文件中。

### 模型测试
你可以通过如下命令测试 Cycle GAN 模型效果：
```bash
python demo.py --model_name your_model_name --dataroot test_img_path
```

如果你只是希望简单查看模型效果，可以执行如下命令：
```bash
python demo.py
```

如果你希望使用自己的图片进行测试，请将图片放在 `./my_img` 目录下，然后执行如下命令：
```bash
python enhance.py
python demo.py
```
你的图片会经过统一的尺寸统一，曝光统一等预处理后，存放在 `./demo_img` 目录下，后续直接使用 `demo.py` 即可进行测试。

所有测试结果会保存在 `./results/model_name/` 目录下。

## 结果分析

### loss 曲线分析
要绘制不同模型的 loss 曲线对比图，你需要首先将不同模型训练得到的 `training_log.txt` 文件改名并存放在 `./log/` 目录下，随后进入`plot.py` 文件，修改 `log_files` 和 `log_names` 两个列表，分别存放不同模型的 log 文件名和模型名称。需修改的代码如下所示：
```python
    log_files = ["log/typical_log.txt", "log/attention_log.txt"]  # 添加你的日志文件路径
    log_names = ["typical", "attention"]  # 日志文件对应的名称
```

最后，执行如下命令即可绘制 loss 曲线对比图：
```bash
python plot.py
```

绘制结果会保存在 `./log/` 目录下。

### 注意力机制效果分析
要对注意力机制进行分析，你首先需要一个训练好的注意力机制模型，并将其保存在 `./checkpoints/` 目录下。然后，你需要进入 `attention.py` 文件，修改 `model_path` 和 `img_path` 两个变量，分别为你的模型存放路径和测试用图片存放路径。需修改的代码如下所示：
```python
    model_path = 'checkpoints/attention/netG_B2A.pth'
    img_path = 'demo_img/0012.png'
```

最后，你可以通过如下命令测试注意力机制效果：
```bash
python attention.py
```

可视化测试结果会被保存在 `attention_visualization.png` 文件中。

## 致谢
- 感谢 [Cycle GAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix?tab=readme-ov-file) 项目对图像风格迁移领域的重要贡献。
- 感谢 DAI RAY 提供的多张优美的摄影作品，用于测试模型效果。