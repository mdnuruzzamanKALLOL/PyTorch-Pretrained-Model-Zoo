<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:EE4C2C,50:FF6B35,100:FF8C00&height=200&section=header&text=PyTorch%20Pretrained%20Model%20Zoo&fontSize=40&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=80%20Production-Ready%20Models%20%7C%20Notebooks%20%7C%20Scripts%20%7C%20Fine-Tuning%20Guides&descAlignY=62&descColor=ffe0b2&descSize=16"/>

<br/>

<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/torchvision-Integrated-FF6B35?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Models-80-4ecdc4?style=for-the-badge&logo=github&logoColor=white"/>
<img src="https://img.shields.io/badge/Families-20-7b2ff7?style=for-the-badge&logo=github&logoColor=white"/>
<img src="https://img.shields.io/badge/ImageNet-Pretrained-success?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-ff6b6b?style=for-the-badge"/>

</div>

---

## What's Inside

Every model folder contains **three ready-to-run resources**:

<div align="center">

| Asset | Description |
|-------|-------------|
| `NoteBook/` | Interactive Jupyter notebook — architecture walkthrough, training & evaluation |
| `Python Scripts/` | Standalone `.py` files — build from scratch, train loop, single-image inference |
| `Using Weight File/` | Scripts to load ImageNet weights, feature-extract (frozen) and fine-tune (progressive unfreeze) |
| `README.md` | Architecture details, layer configs, key specs and references |

</div>

---

<h2 align="center">Model Table</h2>
<div align="center">

### AlexNet

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [AlexNet](Alexnet/) | 61 M | 224² | [📓](Alexnet/NoteBook/) | [🐍](Alexnet/Python%20Scripts/) | [⚖️](Alexnet/Using%20Weight%20File/) |

</div>

<div align="center">

### ConvNeXt Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [ConvNeXt Tiny](ConvNeXt/ConvNeXt%20Tiny/) | 28 M | 224² | [📓](ConvNeXt/ConvNeXt%20Tiny/NoteBook/) | [🐍](ConvNeXt/ConvNeXt%20Tiny/Python%20Scripts/) | [⚖️](ConvNeXt/ConvNeXt%20Tiny/Using%20Weight%20File/) |
| [ConvNeXt Small](ConvNeXt/ConvNeXt%20Small/) | 50 M | 224² | [📓](ConvNeXt/ConvNeXt%20Small/NoteBook/) | [🐍](ConvNeXt/ConvNeXt%20Small/Python%20Scripts/) | [⚖️](ConvNeXt/ConvNeXt%20Small/Using%20Weight%20File/) |
| [ConvNeXt Base](ConvNeXt/ConvNeXt%20Base/) | 89 M | 224² | [📓](ConvNeXt/ConvNeXt%20Base/NoteBook/) | [🐍](ConvNeXt/ConvNeXt%20Base/Python%20Scripts/) | [⚖️](ConvNeXt/ConvNeXt%20Base/Using%20Weight%20File/) |
| [ConvNeXt Large](ConvNeXt/ConvNeXt%20LArge/) | 198 M | 224² | [📓](ConvNeXt/ConvNeXt%20LArge/NoteBook/) | [🐍](ConvNeXt/ConvNeXt%20LArge/Python%20Scripts/) | [⚖️](ConvNeXt/ConvNeXt%20LArge/Using%20Weight%20File/) |

</div>

<div align="center">

### DenseNet Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [DenseNet 121](DenseNet/DenseNet%20121/) | 8 M | 224² | [📓](DenseNet/DenseNet%20121/NoteBook/) | [🐍](DenseNet/DenseNet%20121/Python%20Scripts/) | [⚖️](DenseNet/DenseNet%20121/Using%20Weight%20File/) |
| [DenseNet 161](DenseNet/DenseNet%20161/) | 29 M | 224² | [📓](DenseNet/DenseNet%20161/NoteBook/) | [🐍](DenseNet/DenseNet%20161/Python%20Scripts/) | [⚖️](DenseNet/DenseNet%20161/Using%20Weight%20File/) |
| [DenseNet 169](DenseNet/DenseNet%20169/) | 14 M | 224² | [📓](DenseNet/DenseNet%20169/NoteBook/) | [🐍](DenseNet/DenseNet%20169/Python%20Scripts/) | [⚖️](DenseNet/DenseNet%20169/Using%20Weight%20File/) |
| [DenseNet 201](DenseNet/DenseNet%20201/) | 20 M | 224² | [📓](DenseNet/DenseNet%20201/NoteBook/) | [🐍](DenseNet/DenseNet%20201/Python%20Scripts/) | [⚖️](DenseNet/DenseNet%20201/Using%20Weight%20File/) |

</div>

<div align="center">

### EfficientNet Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [EfficientNet B0](EfficientNet/EfficientNet%20B0/) | 5.3 M | 224² | [📓](EfficientNet/EfficientNet%20B0/NoteBook/) | [🐍](EfficientNet/EfficientNet%20B0/Python%20Scripts/) | [⚖️](EfficientNet/EfficientNet%20B0/Using%20Weight%20File/) |
| [EfficientNet B1](EfficientNet/EfficientNet%20B1/) | 7.8 M | 240² | [📓](EfficientNet/EfficientNet%20B1/NoteBook/) | [🐍](EfficientNet/EfficientNet%20B1/Python%20Scripts/) | [⚖️](EfficientNet/EfficientNet%20B1/Using%20Weight%20File/) |
| [EfficientNet B2](EfficientNet/EfficientNet%20B2/) | 9.1 M | 260² | [📓](EfficientNet/EfficientNet%20B2/NoteBook/) | [🐍](EfficientNet/EfficientNet%20B2/Python%20Scripts/) | [⚖️](EfficientNet/EfficientNet%20B2/Using%20Weight%20File/) |
| [EfficientNet B3](EfficientNet/EfficientNet%20B3/) | 12 M | 300² | [📓](EfficientNet/EfficientNet%20B3/NoteBook/) | [🐍](EfficientNet/EfficientNet%20B3/Python%20Scripts/) | [⚖️](EfficientNet/EfficientNet%20B3/Using%20Weight%20File/) |
| [EfficientNet B4](EfficientNet/EfficientNet%20B4/) | 19 M | 380² | [📓](EfficientNet/EfficientNet%20B4/NoteBook/) | [🐍](EfficientNet/EfficientNet%20B4/Python%20Scripts/) | [⚖️](EfficientNet/EfficientNet%20B4/Using%20Weight%20File/) |
| [EfficientNet B5](EfficientNet/EfficientNet%20B5/) | 30 M | 456² | [📓](EfficientNet/EfficientNet%20B5/NoteBook/) | [🐍](EfficientNet/EfficientNet%20B5/Python%20Scripts/) | [⚖️](EfficientNet/EfficientNet%20B5/Using%20Weight%20File/) |
| [EfficientNet B6](EfficientNet/EfficientNet%20B6/) | 43 M | 528² | [📓](EfficientNet/EfficientNet%20B6/NoteBook/) | [🐍](EfficientNet/EfficientNet%20B6/Python%20Scripts/) | [⚖️](EfficientNet/EfficientNet%20B6/Using%20Weight%20File/) |
| [EfficientNet B7](EfficientNet/EfficientNet%20B7/) | 66 M | 600² | [📓](EfficientNet/EfficientNet%20B7/NoteBook/) | [🐍](EfficientNet/EfficientNet%20B7/Python%20Scripts/) | [⚖️](EfficientNet/EfficientNet%20B7/Using%20Weight%20File/) |

</div>

<div align="center">

### EfficientNetV2 Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [EfficientNetV2 S](EfficientNetV2/EfficientNetV2%20S/) | 21.5 M | 384² | [📓](EfficientNetV2/EfficientNetV2%20S/NoteBook/) | [🐍](EfficientNetV2/EfficientNetV2%20S/Python%20Scripts/) | [⚖️](EfficientNetV2/EfficientNetV2%20S/Using%20Weight%20File/) |
| [EfficientNetV2 M](EfficientNetV2/EfficientNetV2%20M/) | 54.1 M | 480² | [📓](EfficientNetV2/EfficientNetV2%20M/NoteBook/) | [🐍](EfficientNetV2/EfficientNetV2%20M/Python%20Scripts/) | [⚖️](EfficientNetV2/EfficientNetV2%20M/Using%20Weight%20File/) |
| [EfficientNetV2 L](EfficientNetV2/EfficientNetV2%20L/) | 119 M | 480² | [📓](EfficientNetV2/EfficientNetV2%20L/NoteBook/) | [🐍](EfficientNetV2/EfficientNetV2%20L/Python%20Scripts/) | [⚖️](EfficientNetV2/EfficientNetV2%20L/Using%20Weight%20File/) |

</div>

<div align="center">

### GoogLeNet

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [GoogLeNet](GoogLeNet/) | 6.6 M | 224² | [📓](GoogLeNet/NoteBook/) | [🐍](GoogLeNet/Python%20Scripts/) | [⚖️](GoogLeNet/Using%20Weight%20File/) |

</div>

<div align="center">

### Inception V3

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [Inception V3](Inception%20V3/) | 27.2 M | 299² | [📓](Inception%20V3/NoteBook/) | [🐍](Inception%20V3/Python%20Scripts/) | [⚖️](Inception%20V3/Using%20Weight%20File/) |

</div>

<div align="center">

### MaxViT

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [MaxViT T](MaxVit/) | 119 M | 224² | [📓](MaxVit/NoteBook/) | [🐍](MaxVit/Python%20Scripts/) | [⚖️](MaxVit/Using%20Weight%20File/) |

</div>

<div align="center">

### MNASNet Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [MNASNet 0.5](MNASNet/MNASNet%200.5/) | 2.2 M | 224² | [📓](MNASNet/MNASNet%200.5/NoteBook/) | [🐍](MNASNet/MNASNet%200.5/Python%20Scripts/) | [⚖️](MNASNet/MNASNet%200.5/Using%20Weight%20File/) |
| [MNASNet 0.75](MNASNet/MNASNet%200.75/) | 3.2 M | 224² | [📓](MNASNet/MNASNet%200.75/NoteBook/) | [🐍](MNASNet/MNASNet%200.75/Python%20Scripts/) | [⚖️](MNASNet/MNASNet%200.75/Using%20Weight%20File/) |
| [MNASNet 1.0](MNASNet/MNASNet%201.0/) | 4.4 M | 224² | [📓](MNASNet/MNASNet%201.0/NoteBook/) | [🐍](MNASNet/MNASNet%201.0/Python%20Scripts/) | [⚖️](MNASNet/MNASNet%201.0/Using%20Weight%20File/) |
| [MNASNet 1.3](MNASNet/MNASNet%201.3/) | 6.3 M | 224² | [📓](MNASNet/MNASNet%201.3/NoteBook/) | [🐍](MNASNet/MNASNet%201.3/Python%20Scripts/) | [⚖️](MNASNet/MNASNet%201.3/Using%20Weight%20File/) |

</div>

<div align="center">

### MobileNet Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [MobileNet V2](MobileNet%20V2/) | 3.4 M | 224² | [📓](MobileNet%20V2/NoteBook/) | [🐍](MobileNet%20V2/Python%20Scripts/) | [⚖️](MobileNet%20V2/Using%20Weight%20File/) |
| [MobileNet V3 Large](MobileNet%20V3/MobileNet%20V3%20Large/) | 5.5 M | 224² | [📓](MobileNet%20V3/MobileNet%20V3%20Large/NoteBook/) | [🐍](MobileNet%20V3/MobileNet%20V3%20Large/Python%20Scripts/) | [⚖️](MobileNet%20V3/MobileNet%20V3%20Large/Using%20Weight%20File/) |
| [MobileNet V3 Small](MobileNet%20V3/MobileNet%20V3%20Small/) | 2.5 M | 224² | [📓](MobileNet%20V3/MobileNet%20V3%20Small/NoteBook/) | [🐍](MobileNet%20V3/MobileNet%20V3%20Small/Python%20Scripts/) | [⚖️](MobileNet%20V3/MobileNet%20V3%20Small/Using%20Weight%20File/) |

</div>

<div align="center">

### RegNet Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [RegNet X 400MF](RegNet/RegNet%20X_400MF/) | 5.5 M | 224² | [📓](RegNet/RegNet%20X_400MF/NoteBook/) | [🐍](RegNet/RegNet%20X_400MF/Python%20Scripts/) | [⚖️](RegNet/RegNet%20X_400MF/Using%20Weight%20File/) |
| [RegNet X 800MF](RegNet/RegNet%20X_800MF/) | 7.3 M | 224² | [📓](RegNet/RegNet%20X_800MF/NoteBook/) | [🐍](RegNet/RegNet%20X_800MF/Python%20Scripts/) | [⚖️](RegNet/RegNet%20X_800MF/Using%20Weight%20File/) |
| [RegNet X 1.6GF](RegNet/RegNet%20X_1.6GF/) | 9.2 M | 224² | [📓](RegNet/RegNet%20X_1.6GF/NoteBook/) | [🐍](RegNet/RegNet%20X_1.6GF/Python%20Scripts/) | [⚖️](RegNet/RegNet%20X_1.6GF/Using%20Weight%20File/) |
| [RegNet X 3.2GF](RegNet/RegNet%20X_3.2GF/) | 15.3 M | 224² | [📓](RegNet/RegNet%20X_3.2GF/NoteBook/) | [🐍](RegNet/RegNet%20X_3.2GF/Python%20Scripts/) | [⚖️](RegNet/RegNet%20X_3.2GF/Using%20Weight%20File/) |
| [RegNet X 8GF](RegNet/RegNet%20X_8GF/) | 39.6 M | 224² | [📓](RegNet/RegNet%20X_8GF/NoteBook/) | [🐍](RegNet/RegNet%20X_8GF/Python%20Scripts/) | [⚖️](RegNet/RegNet%20X_8GF/Using%20Weight%20File/) |
| [RegNet X 16GF](RegNet/RegNet%20X_16GF/) | 54.3 M | 224² | [📓](RegNet/RegNet%20X_16GF/NoteBook/) | [🐍](RegNet/RegNet%20X_16GF/Python%20Scripts/) | [⚖️](RegNet/RegNet%20X_16GF/Using%20Weight%20File/) |
| [RegNet X 32GF](RegNet/RegNet%20X_32GF/) | 107.8 M | 224² | [📓](RegNet/RegNet%20X_32GF/NoteBook/) | [🐍](RegNet/RegNet%20X_32GF/Python%20Scripts/) | [⚖️](RegNet/RegNet%20X_32GF/Using%20Weight%20File/) |
| [RegNet Y 400MF](RegNet/RegNet%20Y_400MF/) | 4.3 M | 224² | [📓](RegNet/RegNet%20Y_400MF/NoteBook/) | [🐍](RegNet/RegNet%20Y_400MF/Python%20Scripts/) | [⚖️](RegNet/RegNet%20Y_400MF/Using%20Weight%20File/) |
| [RegNet Y 800MF](RegNet/RegNet%20Y_800MF/) | 6.4 M | 224² | [📓](RegNet/RegNet%20Y_800MF/NoteBook/) | [🐍](RegNet/RegNet%20Y_800MF/Python%20Scripts/) | [⚖️](RegNet/RegNet%20Y_800MF/Using%20Weight%20File/) |
| [RegNet Y 1.6GF](RegNet/RegNet%20Y_1.6GF/) | 11.2 M | 224² | [📓](RegNet/RegNet%20Y_1.6GF/NoteBook/) | [🐍](RegNet/RegNet%20Y_1.6GF/Python%20Scripts/) | [⚖️](RegNet/RegNet%20Y_1.6GF/Using%20Weight%20File/) |
| [RegNet Y 3.2GF](RegNet/RegNet%20Y_3.2GF/) | 19.4 M | 224² | [📓](RegNet/RegNet%20Y_3.2GF/NoteBook/) | [🐍](RegNet/RegNet%20Y_3.2GF/Python%20Scripts/) | [⚖️](RegNet/RegNet%20Y_3.2GF/Using%20Weight%20File/) |
| [RegNet Y 8GF](RegNet/RegNet%20Y_8GF/) | 39.4 M | 224² | [📓](RegNet/RegNet%20Y_8GF/NoteBook/) | [🐍](RegNet/RegNet%20Y_8GF/Python%20Scripts/) | [⚖️](RegNet/RegNet%20Y_8GF/Using%20Weight%20File/) |
| [RegNet Y 16GF](RegNet/RegNet%20Y_16GF/) | 83.6 M | 224² | [📓](RegNet/RegNet%20Y_16GF/NoteBook/) | [🐍](RegNet/RegNet%20Y_16GF/Python%20Scripts/) | [⚖️](RegNet/RegNet%20Y_16GF/Using%20Weight%20File/) |
| [RegNet Y 32GF](RegNet/RegNet%20Y_32GF/) | 145 M | 224² | [📓](RegNet/RegNet%20Y_32GF/NoteBook/) | [🐍](RegNet/RegNet%20Y_32GF/Python%20Scripts/) | [⚖️](RegNet/RegNet%20Y_32GF/Using%20Weight%20File/) |
| [RegNet Y 128GF](RegNet/RegNet%20Y_128GF/) | 644.8 M | 224² | [📓](RegNet/RegNet%20Y_128GF/NoteBook/) | [🐍](RegNet/RegNet%20Y_128GF/Python%20Scripts/) | [⚖️](RegNet/RegNet%20Y_128GF/Using%20Weight%20File/) |

</div>

<div align="center">

### ResNet Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [ResNet 18](ResNet/ResNet%2018/) | 11.7 M | 224² | [📓](ResNet/ResNet%2018/NoteBook/) | [🐍](ResNet/ResNet%2018/Python%20Scripts/) | [⚖️](ResNet/ResNet%2018/Using%20Weight%20File/) |
| [ResNet 34](ResNet/ResNet%2034/) | 21.8 M | 224² | [📓](ResNet/ResNet%2034/NoteBook/) | [🐍](ResNet/ResNet%2034/Python%20Scripts/) | [⚖️](ResNet/ResNet%2034/Using%20Weight%20File/) |
| [ResNet 50](ResNet/ResNet%2050/) | 25.6 M | 224² | [📓](ResNet/ResNet%2050/NoteBook/) | [🐍](ResNet/ResNet%2050/Python%20Scripts/) | [⚖️](ResNet/ResNet%2050/Using%20Weight%20File/) |
| [ResNet 101](ResNet/ResNet%20101/) | 44.5 M | 224² | [📓](ResNet/ResNet%20101/NoteBook/) | [🐍](ResNet/ResNet%20101/Python%20Scripts/) | [⚖️](ResNet/ResNet%20101/Using%20Weight%20File/) |
| [ResNet 152](ResNet/ResNet%20152/) | 60.2 M | 224² | [📓](ResNet/ResNet%20152/NoteBook/) | [🐍](ResNet/ResNet%20152/Python%20Scripts/) | [⚖️](ResNet/ResNet%20152/Using%20Weight%20File/) |

</div>

<div align="center">

### ResNeXt Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [ResNeXt 50 32x4D](ResNeXt/ResNeXt%2050_32x4D/) | 25 M | 224² | [📓](ResNeXt/ResNeXt%2050_32x4D/NoteBook/) | [🐍](ResNeXt/ResNeXt%2050_32x4D/Python%20Scripts/) | [⚖️](ResNeXt/ResNeXt%2050_32x4D/Using%20Weight%20File/) |
| [ResNeXt 101 32x8D](ResNeXt/ResNeXt%20101_32x8D/) | 88.8 M | 224² | [📓](ResNeXt/ResNeXt%20101_32x8D/NoteBook/) | [🐍](ResNeXt/ResNeXt%20101_32x8D/Python%20Scripts/) | [⚖️](ResNeXt/ResNeXt%20101_32x8D/Using%20Weight%20File/) |
| [ResNeXt 101 64x4D](ResNeXt/ResNeXt%20101_64x4D/) | 83.5 M | 224² | [📓](ResNeXt/ResNeXt%20101_64x4D/NoteBook/) | [🐍](ResNeXt/ResNeXt%20101_64x4D/Python%20Scripts/) | [⚖️](ResNeXt/ResNeXt%20101_64x4D/Using%20Weight%20File/) |

</div>

<div align="center">

### ShuffleNet V2 Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [ShuffleNet V2 x0.5](ShuffleNet%20V2/ShuffleNet%20V2%20x0.5/) | 1.4 M | 224² | [📓](ShuffleNet%20V2/ShuffleNet%20V2%20x0.5/NoteBook/) | [🐍](ShuffleNet%20V2/ShuffleNet%20V2%20x0.5/Python%20Scripts/) | [⚖️](ShuffleNet%20V2/ShuffleNet%20V2%20x0.5/Using%20Weight%20File/) |
| [ShuffleNet V2 x1.0](ShuffleNet%20V2/ShuffleNet%20V2%20x1.0/) | 2.3 M | 224² | [📓](ShuffleNet%20V2/ShuffleNet%20V2%20x1.0/NoteBook/) | [🐍](ShuffleNet%20V2/ShuffleNet%20V2%20x1.0/Python%20Scripts/) | [⚖️](ShuffleNet%20V2/ShuffleNet%20V2%20x1.0/Using%20Weight%20File/) |
| [ShuffleNet V2 x1.5](ShuffleNet%20V2/ShuffleNet%20V2%20x1.5/) | 3.5 M | 224² | [📓](ShuffleNet%20V2/ShuffleNet%20V2%20x1.5/NoteBook/) | [🐍](ShuffleNet%20V2/ShuffleNet%20V2%20x1.5/Python%20Scripts/) | [⚖️](ShuffleNet%20V2/ShuffleNet%20V2%20x1.5/Using%20Weight%20File/) |
| [ShuffleNet V2 x2.0](ShuffleNet%20V2/ShuffleNet%20V2%20x2.0/) | 7.4 M | 224² | [📓](ShuffleNet%20V2/ShuffleNet%20V2%20x2.0/NoteBook/) | [🐍](ShuffleNet%20V2/ShuffleNet%20V2%20x2.0/Python%20Scripts/) | [⚖️](ShuffleNet%20V2/ShuffleNet%20V2%20x2.0/Using%20Weight%20File/) |

</div>

<div align="center">

### SqueezeNet Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [SqueezeNet 1.0](SqueezeNet/SqueezeNet%201.0/) | 1.2 M | 224² | [📓](SqueezeNet/SqueezeNet%201.0/NoteBook/) | [🐍](SqueezeNet/SqueezeNet%201.0/Python%20Scripts/) | [⚖️](SqueezeNet/SqueezeNet%201.0/Using%20Weight%20File/) |
| [SqueezeNet 1.1](SqueezeNet/SqueezeNet%201.1/) | 1.2 M | 224² | [📓](SqueezeNet/SqueezeNet%201.1/NoteBook/) | [🐍](SqueezeNet/SqueezeNet%201.1/Python%20Scripts/) | [⚖️](SqueezeNet/SqueezeNet%201.1/Using%20Weight%20File/) |

</div>

<div align="center">

### Swin Transformer Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [Swin-T](SwinTransformer/Swin_T/) | 28 M | 224² | [📓](SwinTransformer/Swin_T/NoteBook/) | [🐍](SwinTransformer/Swin_T/Python%20Scripts/) | [⚖️](SwinTransformer/Swin_T/Using%20Weight%20File/) |
| [Swin-S](SwinTransformer/Swin_S/) | 50 M | 224² | [📓](SwinTransformer/Swin_S/NoteBook/) | [🐍](SwinTransformer/Swin_S/Python%20Scripts/) | [⚖️](SwinTransformer/Swin_S/Using%20Weight%20File/) |
| [Swin-B](SwinTransformer/Swin_B/) | 88 M | 224² | [📓](SwinTransformer/Swin_B/NoteBook/) | [🐍](SwinTransformer/Swin_B/Python%20Scripts/) | [⚖️](SwinTransformer/Swin_B/Using%20Weight%20File/) |
| [Swin-V2-T](SwinTransformer/Swin_V2_T/) | 28 M | 256² | [📓](SwinTransformer/Swin_V2_T/NoteBook/) | [🐍](SwinTransformer/Swin_V2_T/Python%20Scripts/) | [⚖️](SwinTransformer/Swin_V2_T/Using%20Weight%20File/) |
| [Swin-V2-S](SwinTransformer/Swin_V2_S/) | 50 M | 256² | [📓](SwinTransformer/Swin_V2_S/NoteBook/) | [🐍](SwinTransformer/Swin_V2_S/Python%20Scripts/) | [⚖️](SwinTransformer/Swin_V2_S/Using%20Weight%20File/) |
| [Swin-V2-B](SwinTransformer/Swin_V2_B/) | 88 M | 256² | [📓](SwinTransformer/Swin_V2_B/NoteBook/) | [🐍](SwinTransformer/Swin_V2_B/Python%20Scripts/) | [⚖️](SwinTransformer/Swin_V2_B/Using%20Weight%20File/) |

</div>

<div align="center">

### VGG Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [VGG 11](VGG/VGG%2011/) | 133 M | 224² | [📓](VGG/VGG%2011/NoteBook/) | [🐍](VGG/VGG%2011/Python%20Scripts/) | [⚖️](VGG/VGG%2011/Using%20Weight%20File/) |
| [VGG 11 BN](VGG/VGG%2011_BN/) | 133 M | 224² | [📓](VGG/VGG%2011_BN/NoteBook/) | [🐍](VGG/VGG%2011_BN/Python%20Scripts/) | [⚖️](VGG/VGG%2011_BN/Using%20Weight%20File/) |
| [VGG 13](VGG/VGG%2013/) | 133 M | 224² | [📓](VGG/VGG%2013/NoteBook/) | [🐍](VGG/VGG%2013/Python%20Scripts/) | [⚖️](VGG/VGG%2013/Using%20Weight%20File/) |
| [VGG 13 BN](VGG/VGG%2013_BN/) | 133 M | 224² | [📓](VGG/VGG%2013_BN/NoteBook/) | [🐍](VGG/VGG%2013_BN/Python%20Scripts/) | [⚖️](VGG/VGG%2013_BN/Using%20Weight%20File/) |
| [VGG 16](VGG/VGG%2016/) | 138 M | 224² | [📓](VGG/VGG%2016/NoteBook/) | [🐍](VGG/VGG%2016/Python%20Scripts/) | [⚖️](VGG/VGG%2016/Using%20Weight%20File/) |
| [VGG 16 BN](VGG/VGG%2016_BN/) | 138 M | 224² | [📓](VGG/VGG%2016_BN/NoteBook/) | [🐍](VGG/VGG%2016_BN/Python%20Scripts/) | [⚖️](VGG/VGG%2016_BN/Using%20Weight%20File/) |
| [VGG 19](VGG/VGG%2019/) | 144 M | 224² | [📓](VGG/VGG%2019/NoteBook/) | [🐍](VGG/VGG%2019/Python%20Scripts/) | [⚖️](VGG/VGG%2019/Using%20Weight%20File/) |
| [VGG 19 BN](VGG/VGG%2019_BN/) | 144 M | 224² | [📓](VGG/VGG%2019_BN/NoteBook/) | [🐍](VGG/VGG%2019_BN/Python%20Scripts/) | [⚖️](VGG/VGG%2019_BN/Using%20Weight%20File/) |

</div>

<div align="center">

### Vision Transformer Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [ViT-B/16](VisionTransformer/ViT%20b_16/) | 86 M | 224² | [📓](VisionTransformer/ViT%20b_16/NoteBook/) | [🐍](VisionTransformer/ViT%20b_16/Python%20Scripts/) | [⚖️](VisionTransformer/ViT%20b_16/Using%20Weight%20File/) |
| [ViT-B/32](VisionTransformer/ViT%20b_32/) | 88 M | 224² | [📓](VisionTransformer/ViT%20b_32/NoteBook/) | [🐍](VisionTransformer/ViT%20b_32/Python%20Scripts/) | [⚖️](VisionTransformer/ViT%20b_32/Using%20Weight%20File/) |
| [ViT-L/16](VisionTransformer/ViT%20l_16/) | 307 M | 224² | [📓](VisionTransformer/ViT%20l_16/NoteBook/) | [🐍](VisionTransformer/ViT%20l_16/Python%20Scripts/) | [⚖️](VisionTransformer/ViT%20l_16/Using%20Weight%20File/) |
| [ViT-L/32](VisionTransformer/ViT%20l_32/) | 307 M | 224² | [📓](VisionTransformer/ViT%20l_32/NoteBook/) | [🐍](VisionTransformer/ViT%20l_32/Python%20Scripts/) | [⚖️](VisionTransformer/ViT%20l_32/Using%20Weight%20File/) |
| [ViT-H/14](VisionTransformer/ViT%20h_14/) | 633 M | 518² | [📓](VisionTransformer/ViT%20h_14/NoteBook/) | [🐍](VisionTransformer/ViT%20h_14/Python%20Scripts/) | [⚖️](VisionTransformer/ViT%20h_14/Using%20Weight%20File/) |

</div>

<div align="center">

### Wide ResNet Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [Wide ResNet 50-2](Wide%20ResNet/Wide%20ResNet%2050_2/) | 68.9 M | 224² | [📓](Wide%20ResNet/Wide%20ResNet%2050_2/NoteBook/) | [🐍](Wide%20ResNet/Wide%20ResNet%2050_2/Python%20Scripts/) | [⚖️](Wide%20ResNet/Wide%20ResNet%2050_2/Using%20Weight%20File/) |
| [Wide ResNet 101-2](Wide%20ResNet/Wide%20ResNet%20101_2/) | 126.9 M | 224² | [📓](Wide%20ResNet/Wide%20ResNet%20101_2/NoteBook/) | [🐍](Wide%20ResNet/Wide%20ResNet%20101_2/Python%20Scripts/) | [⚖️](Wide%20ResNet/Wide%20ResNet%20101_2/Using%20Weight%20File/) |

</div>

---

## Quick Start

### Train from scratch

```bash
cd "ResNet/ResNet 50/Python Scripts"
python train.py
```

### Fine-tune with pretrained ImageNet weights

```bash
cd "EfficientNet/EfficientNet B0/Using Weight File"
python fine_tuning.py
```

### Feature extraction (frozen base)

```bash
cd "VGG/VGG 16/Using Weight File"
python feature_extraction.py
```

---

<div align="center">

## Families at a Glance

| Family | Models | Best Use |
|--------|:------:|----------|
| **AlexNet** | 1 | Historical baseline, fast training |
| **ConvNeXt** | 4 | Modern CNN, ViT-competitive accuracy |
| **DenseNet** | 4 | Dense skip connections, feature reuse |
| **EfficientNet** | 8 | Compound-scaled, accuracy/efficiency |
| **EfficientNetV2** | 3 | Faster training with Fused-MBConv |
| **GoogLeNet** | 1 | Inception modules, multi-scale features |
| **Inception V3** | 1 | Multi-scale, medical imaging |
| **MaxViT** | 1 | Hybrid CNN + ViT with local-global attention |
| **MNASNet** | 4 | NAS-designed, mobile-optimized |
| **MobileNet** | 3 | Edge & mobile deployment |
| **RegNet** | 15 | Systematic design, scalable efficiency |
| **ResNet** | 5 | Backbone standard, easy fine-tuning |
| **ResNeXt** | 3 | Grouped convolutions, stronger ResNet |
| **ShuffleNet V2** | 4 | Ultra-lightweight, mobile inference |
| **SqueezeNet** | 2 | Minimal params, tiny footprint |
| **Swin Transformer** | 6 | Shifted-window attention, hierarchical |
| **VGG** | 8 | Simple baseline, style transfer |
| **Vision Transformer** | 5 | Pure attention, patch-based |
| **Wide ResNet** | 2 | Wider channels, improved ResNet |

</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.PyTorch-Pretrained-Model-Zoo&left_color=%23EE4C2C&right_color=%230e75b6&left_text=Page%20Views)

</div>

<div align="center">
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:FF8C00,50:FF6B35,100:EE4C2C&height=120&section=footer&text=Happy%20Training!%20%F0%9F%94%A5&fontSize=24&fontColor=ffffff&animation=fadeIn&fontAlignY=65"/>
</div>

