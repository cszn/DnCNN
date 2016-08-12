## Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising


### Main Contents

**demos**:  `Demo_test_DnCNN-.m`.

**model**:  including the trained models for Gaussian denoising; a single model for Gaussian denoising, single image super-resolution (SISR) and deblocking.

**testsets**:  BSD68 and Set10 for Gaussian denoising evaluation; Set5, Set14, BSD100 and Urban100 datasets for SISR evaluation; Classic5 and LIVE1 for JPEG image deblocking evaluation.



To run the testing demos `Demo_test_DnCNN-.m`, you should first [install](http://www.vlfeat.org/matconvnet/install/) [MatConvNet](http://www.vlfeat.org/matconvnet/).

Note: If you did not install MatConvNet, just replace `res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test')` with `res = simplenn_matlab(net, input)`.

For the training code, feel free to contact: cskaizhang@gmail.com



### Results

#### Gaussian Denoising

The average PSNR(dB) results of different methods on the BSD6868 dataset.

|  Noise Level | BM3D | WNNM  | EPLL | MLP |  CSF |TNRD  | DnCNN-S | DnCNN-B |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| 15  |  31.07  |   31.37   | 31.21  |   -   |  31.24 |  31.42 | **31.73** | **31.61**  |
| 25  |  28.57  |   28.83   | 28.68  | 28.96 |  28.74 |  28.92 | **29.23** | **29.16**  |
| 50  |  25.62  |   25.87   | 25.67  | 26.03 |    -   |  25.97 | **26.23** | **26.23**  |

#### Gaussian Denoising, Single ImageSuper-Resolution and JPEG Image Deblocking via a Single (DnCNN-3) Model 

Average PSNR(dB)/SSIM results of different methods for Gaussian denoising with noise level 15, 25 and 50 on BSD68 dataset, single image super-resolution with 
upscaling factors 2, 3 and 40 on Set5, Set14, BSD100 and Urban100 datasets, JPEG image deblocking with quality factors 10, 20, 30 and 40 on Classic5 and LIVE11 datasets.

###### Gaussian Denoising
|  Dataset    | Noise Level | BM3D | TNRD | DnCNN-3 |
|:---------:|:---------:|:---------:|:---------:|:---------:|
|       |  15  | 31.08 / 0.8722 | 31.42 / 0.8826 | 31.46 / 0.8826 |
| BSD68 |  25  | 28.57 / 0.8017 | 28.92 / 0.8157 | 29.02 / 0.8190 |
|       |  50  | 25.62 / 0.6869 | 25.97 / 0.7029 | 26.10 / 0.7076 |
###### Single Image Super-Resolution
| Dataset | Upscaling Factor | TNRD | VDSR |DnCNN-3|
|:---------:|:---------:|:---------:|:---------:|:---------:|
|        | 2 | 36.86 / 0.9556 | 37.56 / 0.9591 | 37.58 / 0.9590 |
|Set5    | 3 | 33.18 / 0.9152 | 33.67 / 0.9220 | 33.75 / 0.9222 |
|        | 4 | 30.85 / 0.8732 | 31.35 / 0.8845 | 31.40 / 0.8845 |
||
|        | 2 | 32.51 / 0.9069 | 33.02 / 0.9128 | 33.03 / 0.9128 |
|Set14   | 3 | 29.43 / 0.8232 | 29.77 / 0.8318 | 29.81 / 0.8321 |
|        | 4 | 27.66 / 0.7563 | 27.99 / 0.7659 | 28.04 / 0.7672 |
||
|        | 2 | 31.40 / 0.8878 | 31.89 / 0.8961 | 31.90 / 0.8961 |
|BSD100  | 3 | 28.50 / 0.7881 | 28.82 / 0.7980 | 28.85 / 0.7981 |
|        | 4 | 27.00 / 0.7140 | 27.28 / 0.7256 | 27.29 / 0.7253 |
||
|        | 2 | 29.70 / 0.8994 | 30.76 / 0.9143 | 30.74 / 0.9139 |
|Urban100| 3 | 26.42 / 0.8076 | 27.13 / 0.8283 | 27.15 / 0.8276 |
|        | 4 | 24.61 / 0.7291 | 25.17 / 0.7528 | 25.20 / 0.7521 |
###### JPEG Image Deblocking
|  Dataset | Quality Factor | AR-CNN | TNRD | DnCNN-3 |
|:---------:|:---------:|:---------:|:---------:|:---------:|
|Classic5| 10 | 29.03 / 0.7929 | 29.28 / 0.7992 | 29.40 / 0.8026 |
|        | 20 | 31.15 / 0.8517 | 31.47 / 0.8576 | 31.63 / 0.8610 |
|        | 30 | 32.51 / 0.8806 | 32.78 / 0.8837 | 32.91 / 0.8861 |
|        | 40 | 33.34 / 0.8953 |       -        | 33.77 / 0.9003 |
||
|  LIVE1 | 10 | 28.96 / 0.8076 | 29.15 / 0.8111 | 29.19 / 0.8123 |
|        | 20 | 31.29 / 0.8733 | 31.46 / 0.8769 | 31.59 / 0.8802 |
|        | 30 | 32.67 / 0.9043 | 32.84 / 0.9059 | 32.98 / 0.9090 |
|        | 40 | 33.63 / 0.9198 |       -        | 33.96 / 0.9247 |