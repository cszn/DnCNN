
% ======================================================================
% This is the training demo of the paper "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"

% Version:       1.0 (18/08/2016)
% Contact:       Kai Zhang (e-mail: cskaizhang@gmail.com)

% ----------------------------------------------------------------------
% Please consider the following citation if this code is useful to you.

@article{zhang2017beyond,
   title={Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising},
   author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
   journal={IEEE Transactions on Image Processing}, 
   year={2017}, 
   volume={PP}, 
   number={99}, 
   pages={1-1}, 
   doi={10.1109/TIP.2017.2662206},
 }


% ------ Contents -----------------------------------------------------
data
----Test (Set12 and Set68)
----Train400 (400 training images of size 180X180)
----utilities 
----GenerateData_model_64_25_Res_Bnorm_Adam.m (run this to generate training patches!)
Demo_Test_model_64_25_Res_Bnorm_Adam.m   (test each model)
Demo_Train_model_64_25_Res_Bnorm_Adam.m  (run this to train the model)
DnCNN_init_model_64_25_Res_Bnorm_Adam.m  (initializate the model)
DnCNN_train.m (the main body of training code)
vl_nnloss.m   (loss function)
README.txt
% ----------------------------------------------------------------------

% ----------------------------------------------------------------------
% I have tried to make the code as simple as possible. You can change
% 'cnn_train.m' or 'cnn_train_dag.m' in Matconvnet package if necessary.
% ----------------------------------------------------------------------

% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
% ----------------------------------------------------------------------
% If you find any bug, please contact cskaizhang@gmail.com.
% ----------------------------------------------------------------------




