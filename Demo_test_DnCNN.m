
%%% This is the testing demo for gray image (Gaussian) denoising.
%%% Training data: 400 images of size 180X180


% clear; clc;
addpath('utilities');
folderTest  = fullfile('testsets','Set12'); %%% test dataset
%folderTest  = 'testsets\BSD68';
folderModel = 'model';
noiseSigma  = 25;  %%% image noise level
showResult  = 1;
useGPU      = 1;
pauseTime   = 1;

%%% load [specific] Gaussian denoising model

modelSigma  = min(75,max(10,round(noiseSigma/5)*5)); %%% model noise level
load(fullfile(folderModel,'specifics',['sigma=',num2str(modelSigma,'%02d'),'.mat']));

%%% load [blind] Gaussian denoising model %%% for sigma in [0,55]

% load(fullfile(folderModel,'GD_Gray_Blind.mat'));


%%%
net = vl_simplenn_tidy(net);

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
SSIMs = zeros(1,length(filePaths));

for i = 1:length(filePaths)
    
    %%% read images
    label = imread(fullfile(folderTest,filePaths(i).name));
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    label = im2double(label);
    
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
    
    %%% calculate PSNR and SSIM
    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
    if showResult
        imshow(cat(2,im2uint8(label),im2uint8(input),im2uint8(output)));
        title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        drawnow;
        pause(pauseTime)
    end
    PSNRs(i) = PSNRCur;
    SSIMs(i) = SSIMCur;
end

disp([mean(PSNRs),mean(SSIMs)]);


