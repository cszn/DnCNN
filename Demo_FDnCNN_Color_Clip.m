% This is the testing demo of Flexible DnCNN (FDnCNN) for denoising noisy color images corrupted by
% AWGN with clipping setting. The noisy input is 8-bit quantized.
%
% To run the code, you should install Matconvnet first. Alternatively, you can use the
% function `vl_ffdnet_matlab` to perform denoising without Matconvnet.
%
% "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
% "FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising"
%
% Denoising" 2018/05
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)

% clear; clc;

format compact;
global sigmas; % input noise level or input noise level map
addpath(fullfile('utilities'));

folderModel = 'model';
folderTest  = 'testsets';
folderResult= 'results';
imageSets   = {'CBSD68','Kodak24','McMaster'}; % testing datasets
setTestCur  = imageSets{1};      % current testing dataset

showResult  = 1;
useGPU      = 1;
pauseTime   = 0;

imageNoiseSigma = 25;  % image noise level, 25.5 is the default setting of imnoise( ,'gaussian')
inputNoiseSigma = 25;  % input noise level

folderResultCur       =  fullfile(folderResult, [setTestCur,'_Clip_',num2str(imageNoiseSigma(1)),'_',num2str(inputNoiseSigma(1))]);
if ~isdir(folderResultCur)
    mkdir(folderResultCur)
end

load(fullfile('model','FDnCNN_Clip_color.mat'));
net = vl_simplenn_tidy(net);

% for i = 1:size(net.layers,2)
%     net.layers{i}.precious = 1;
% end

if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

% read images
ext         =  {'*.jpg','*.png','*.bmp','*.tif'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,setTestCur,ext{i})));
end

% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));

for i = 1:length(filePaths)
    
    % read images
    label   = imread(fullfile(folderTest,setTestCur,filePaths(i).name));
    [w,h,c] = size(label);
    
    if c == 3
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        label = im2single(label);
        
        % add noise
        randn('seed',0);
        %input = imnoise(label,'gaussian'); % corresponds to imageNoiseSigma = 25.5;
        input = imnoise(label,'gaussian',0,(imageNoiseSigma/255)^2);
        
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
        %toc;
        
        % calculate PSNR, SSIM and save results
        [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
        if showResult
            imshow(cat(2,im2uint8(input),im2uint8(label),im2uint8(output)));
            title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
            % imwrite(im2uint8(output), fullfile(folderResultCur, [nameCur, '_' num2str(imageNoiseSigma(1),'%02d'),'_' num2str(inputNoiseSigma(1),'%02d'),'_PSNR_',num2str(PSNRCur*100,'%4.0f'), extCur] ));
            % imwrite(im2uint8(input), fullfile(folderResultCur, [nameCur, '_' num2str(imageNoiseSigma(1),'%02d'),'_' num2str(inputNoiseSigma(1),'%02d'), extCur] ));
            drawnow;
            pause(pauseTime)
        end
        disp([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        PSNRs(i) = PSNRCur;
        SSIMs(i) = SSIMCur;
        
    end
end

disp([mean(PSNRs),mean(SSIMs)]);




