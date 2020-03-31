# Video Classificaiton

## Overview

- UCF-101より10クラスを対象に動画分類を行った
- 実装はKeras
  - 3DCNN
  - C3D

## Result

- 3DCNN

![3DCNN acc](./Result/01/3DCNN/Batch8_Epoch20_LR0_0001_acc.png)![3DCNN loss](./Result/01/3DCNN/Batch8_Epoch20_LR0_0001_loss.png)

- C3D

![C3D acc](./Result/01/C3D/Batch8_Epoch20_LR0_0001_acc.png)![C3D loss](./Result/01/C3D/Batch8_Epoch20_LR0_0001_loss.png)

## Require

### Enviroments

- Windows10
- Python 3.7.3

### Libraly

- Keras==2.3.1
- matplotlib==3.1.3
- opencv-python==4.2.0.32
- Pillow==7.0.0
- scikit-learn==0.22.2
- Tensorflow-gpu==2.0.0
- tqdm==4.43.0
