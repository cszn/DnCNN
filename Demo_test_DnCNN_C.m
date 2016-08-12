
%%% This is the testing code demo for color image (Gaussian) denoising.
%%% The model is trained with 1) noise levels in [0 55]; 2) 432 training images.


% clear; clc;
addpath('utilities');
folderTest  = 'testsets\CBSD68'; %%% test dataset
folderModel = 'model';
noiseSigma  = 45;  %%% image noise level
showResult  = 1;
useGPU      = 1;
pauseTime   = 1;


%%% load blind Gaussian denoising model (color image)
load(fullfile(folderModel,'GD_Color_Blind.mat')); %%% for sigma in [0,55]

%%%
% net = vl_simplenn_tidy(net);

% for i = 1:size(net.layers,2)
%     net.layers{i}.precious = 1;
% end

%%% move to gpu
if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

%%% read images
ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
end

%%% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));

for i = 1:length(filePaths)
    
    %%% read current image
    label = imread(fullfile(folderTest,filePaths(i).name));
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    label = im2double(label);
    
    %%% add Gaussian noise
    randn('seed',0);
    input = single(label + noiseSigma/255*randn(size(label)));
    
    %%% convert to GPU
    if useGPU
        input = gpuArray(input);
    end
    
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
    %res = simplenn_matlab(net, input); %%% use this if you did not install matconvnet.
    output = input - res(end).x;
    
    %%% convert to CPU
    if useGPU
        output = gather(output);
        input  = gather(input);
    end
    
    %%% calculate PSNR
    [PSNRCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
    if showResult
        imshow(cat(2,im2uint8(label),im2uint8(input),im2uint8(output)));
        title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB'])
        drawnow;
        pause(pauseTime)
    end
    PSNRs(i) = PSNRCur;
    
end

disp(mean(PSNRs));


