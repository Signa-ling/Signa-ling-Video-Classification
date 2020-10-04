# Video Classificaiton

## Overview

- Video classification was performed for 10 classes from UCF-101.
- Keras is used for implementation and Tensorflow is used for the backend.
- The following models are implemented.
  - 3DCNN
  - C3D

- Data breakdown
  - Height×Width : 112×112
  - Color: 1channel (=monochrome)
  - Training data: 985(Train on 788, validate on 197)
  - Test data: 389

## Result

### 3DCNN

- 20epoch

![3DCNN acc 20epoch](./Result/01/3DCNN/Batch8_Epoch20_LR0_0001_acc.png)![3DCNN loss 20epoch](./Result/01/3DCNN/Batch8_Epoch20_LR0_0001_loss.png)

```evalute.py
Test loss:  1.15104304780767
Test accuracy:  0.6323907375335693
```

---

### C3D

- 20epoch

![C3D acc 20epoch](./Result/01/C3D/Batch8_Epoch20_LR0_0001_acc.png)![C3D loss 20epoch](./Result/01/C3D/Batch8_Epoch20_LR0_0001_loss.png)

```evalute.py
Test loss:  4.000657923423845
Test accuracy:  0.596401035785675
```

---

### LSTM

![LSTM acc 20 epoch](./Result/01/LSTM/Batch8_Epoch20_LR0.0001_acc.png)![LSTM loss 20epoch](./Result/01/LSTM/Batch8_Epoch20_LR0.0001_loss.png)

```evalute.py
Test loss:  1.93994140625
Test accuracy:  0.570694088935852
```

## Require

### Enviroments

- Windows10
- Python 3.7.3

### Libraly

See "requirements.txt".
