% This is the testing demo of Flexible DnCNN (FDnCNN) for denoising noisy grayscale images corrupted by
% AWGN.
%
% To run the code, you should install Matconvnet first. Alternatively, you can use the
% function `vl_ffdnet_matlab` to perform denoising without Matconvnet.
%
% "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
% "FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising"
%
%  2018/05
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)

%clear; clc;
format compact;
global sigmas; % input noise level or input noise level map
addpath(fullfile('utilities'));

folderModel = 'model';
folderTest  = 'testsets';
folderResult= 'results';
imageSets   = {'BSD68','Set12'}; % testing datasets
setTestCur  = imageSets{2};      % current testing dataset

showResult  = 1;
useGPU      = 1; % CPU or GPU. For single-threaded (ST) CPU computation, use "matlab -singleCompThread" to start matlab.
pauseTime   = 0;

imageNoiseSigma = 50;  % image noise level
inputNoiseSigma = 50;  % input noise level

folderResultCur       =  fullfile(folderResult, [setTestCur,'_',num2str(imageNoiseSigma),'_',num2str(inputNoiseSigma)]);
if ~isdir(folderResultCur)
    mkdir(folderResultCur)
end

load(fullfile('model','FDnCNN_gray.mat'));
net = vl_simplenn_tidy(net);

% for i = 1:size(net.layers,2)
%     net.layers{i}.precious = 1;
% end

if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

% read images
ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,setTestCur,ext{i})));
end

% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));

for i = 1:length(filePaths)
    
    % read images
    label = imread(fullfile(folderTest,setTestCur,filePaths(i).name));
    [w,h,~]=size(label);
    if size(label,3)==3
        label = rgb2gray(label);
    end
    
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    label = im2double(label);
    
    % add noise
    randn('seed',0);
    noise = imageNoiseSigma/255.*randn(size(label));
    input = single(label + noise);
    
    % tic;
    if useGPU
        input = gpuArray(input);
    end
    
    % set noise level map
    sigmas = inputNoiseSigma/255; % see "vl_simplenn.m".
    
    % perform denoising
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default
    %res    = vl_ffdnet_concise(net, input);    % concise version of vl_simplenn for testing FFDNet
    %res    = vl_ffdnet_matlab(net, input); % use this if you did  not install matconvnet; very slow . note: you should also comment net = vl_simplenn_tidy(net); and if useGPU net = vl_simplenn_move(net, 'gpu') ; end
    
    output = res(end).x;
    
    if useGPU
        output = gather(output);
        input  = gather(input);
    end
    % toc;
    % calculate PSNR, SSIM and save results
    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
    if showResult
        imshow(cat(2,im2uint8(input),im2uint8(label),im2uint8(output)));
        title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        %imwrite(im2uint8(output), fullfile(folderResultCur, [nameCur, '_' num2str(imageNoiseSigma,'%02d'),'_' num2str(inputNoiseSigma,'%02d'),'_PSNR_',num2str(PSNRCur*100,'%4.0f'), extCur] ));
        drawnow;
        pause(pauseTime)
    end
    disp([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
    PSNRs(i) = PSNRCur;
    SSIMs(i) = SSIMCur;
end

disp([mean(PSNRs),mean(SSIMs)]);




