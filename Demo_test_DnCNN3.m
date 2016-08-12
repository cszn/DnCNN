
%%% This is the testing demo for learning a single model for three tasks, including Gaussian denoing, SISR, JPEG image deblocking.

% clear; clc;

addpath('utilities');

%%% testing set
tasks       = {'GD','SR','DB'}; %%% three tasks
imageSets   = {'BSD68','Set5','Set14','BSD100','Urben100','classic5','LIVE1'}; %%% testing dataset

%%% setting
taskTest    = tasks([1 2 3]); %%% choose the tasks for evaluation
setTest     = {imageSets([1]),imageSets([2:5]),imageSets([6 7])}; %%% select the datasets for each tasks
showResult  = [1 1 1]; %%% save the restored images
pauseTime   = 1;
folderModel = 'model';
useGPU      = 1; % 1 or 0, true or false

folderTest  = 'testsets';
folderResult= 'results';

if ~exist(folderResult,'file')
    mkdir(folderResult);
end

%%% task GD = Gaussian Denoising
sigma   = 25;

%%% task SR = Single Image Super-Resolution
scale   = 3;

%%% task DB = DeBlocking
Q       = 20;

%%% load DnCNN-3 model
load(fullfile(folderModel,'DnCNN3.mat'));

%net = vl_simplenn_tidy(net);
% for i = 1:size(net.layers,2)
%     net.layers{i}.precious = 1;
% end
if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

%%% input (single); output (single); label (ground-truth, uint8)
%%% input_RGB (uint8); output_RGB (uint8); label_RGB (ground-truth, uint8)

%%%-------------------------------------------------------------------------------------
%%% Gaussian Denoising (GD)
%%%-------------------------------------------------------------------------------------

if ismember('GD',taskTest)
    taskTestCur = 'GD';
    for n_set = 1 : numel(setTest{1})
        %%% read images
        setTestCur = cell2mat(setTest{1}(n_set));
        disp('-----------------------------------------------');
        disp(['----',setTestCur,'------Gaussian Denoising-----']);
        disp('-----------------------------------------------');
        
        folderTestCur = fullfile(folderTest,setTestCur);
        ext                 =  {'*.jpg','*.png','*.bmp'};
        filepaths           =  [];
        for i = 1 : length(ext)
            filepaths = cat(1,filepaths,dir(fullfile(folderTestCur, ext{i})));
        end
        
        eval(['PSNR_',taskTestCur,'_',setTestCur,'_s',num2str(sigma),' = zeros(length(filepaths),1);']);
        eval(['SSIM_',taskTestCur,'_',setTestCur,'_s',num2str(sigma),' = zeros(length(filepaths),1);']);
        
        %%% folder to store results
        folderResultCur = fullfile(folderResult, [taskTestCur,'_',setTestCur,'_s',num2str(sigma)]);
        if ~exist(folderResultCur,'file')
            mkdir(folderResultCur);
        end
        
        for i = 1 : length(filepaths)
            label  = imread(fullfile(folderTestCur,filepaths(i).name));
            [~,imageName,ext] = fileparts(filepaths(i).name);
            chanel = size(label,3);
            if chanel == 3
                %%% label (uint8)
                label = rgb2gray(label);
            end
            %%% input (single)
            randn('seed',0);
            input = single(im2double(label) + sigma/255*randn(size(label)));
            
            if useGPU
                input = gpuArray(input);
            end
            
            res = vl_simplenn(net, input,[],[],'conserveMemory',true,'mode','test');
            im = res(end).x;
            
            %%% output (single)
            output = gather(input - im);
            
            [PSNR_Cur,SSIM_Cur] = Cal_PSNRSSIM(label,im2uint8(output),0,0);
            disp(['Denoising     ',num2str(PSNR_Cur,'%2.2f'),'dB','    ',filepaths(i).name]);
            eval(['PSNR_',taskTestCur,'_',setTestCur,'_s',num2str(sigma),'(',num2str(i),') = PSNR_Cur;']);
            eval(['SSIM_',taskTestCur,'_',setTestCur,'_s',num2str(sigma),'(',num2str(i),') = SSIM_Cur;']);
            if showResult(1)
                imshow(cat(1,cat(2,im2uint8(input),im2uint8(output)),cat(2,im2uint8(abs(input-output)*10),label)));
                drawnow;
                title(['Denoising     ',filepaths(i).name,'    ',num2str(PSNR_Cur,'%2.2f'),'dB'],'FontSize',12)
                pause(pauseTime)
                %pause()
                
                %%% save results
                imwrite(output,fullfile(folderResultCur,[imageName,'_s',num2str(sigma),'.png']));
                
            end
            
        end
        disp(['Average PSNR is ',num2str(mean(eval(['PSNR_',taskTestCur,'_',setTestCur,'_s',num2str(sigma)])),'%2.2f'),'dB']);
        disp(['Average SSIM is ',num2str(mean(eval(['SSIM_',taskTestCur,'_',setTestCur,'_s',num2str(sigma)])),'%2.4f')]);
        
        %%% save PSNR and SSIM metrics
        save(fullfile(folderResultCur,['PSNR_',taskTestCur,'_',setTestCur,'_s',num2str(sigma),'.mat']),['PSNR_',taskTestCur,'_',setTestCur,'_s',num2str(sigma)])
        save(fullfile(folderResultCur,['SSIM_',taskTestCur,'_',setTestCur,'_s',num2str(sigma),'.mat']),['SSIM_',taskTestCur,'_',setTestCur,'_s',num2str(sigma)])
        
    end
end

%%%-------------------------------------------------------------------------------------
%%% Single Image Super-Resolution (SR)
%%%-------------------------------------------------------------------------------------

if ismember('SR',taskTest)
    taskTestCur = 'SR';
    for n_set = 1 : numel(setTest{2})
        %%% read images
        setTestCur = cell2mat(setTest{2}(n_set));
        disp('--------------------------------------------');
        disp(['----',setTestCur,'-----Super-Resolution-----']);
        disp('--------------------------------------------');
        folderTestCur = fullfile(folderTest,setTestCur);
        ext                 =  {'*.jpg','*.png','*.bmp'};
        filepaths           =  [];
        for i = 1 : length(ext)
            filepaths = cat(1,filepaths,dir(fullfile(folderTestCur, ext{i})));
        end
        eval(['PSNR_',taskTestCur,'_',setTestCur,'_x',num2str(scale),' = zeros(length(filepaths),1);']);
        eval(['SSIM_',taskTestCur,'_',setTestCur,'_x',num2str(scale),' = zeros(length(filepaths),1);']);
        
        if fix(scale) == scale
            crop = scale;
        else
            crop = scale*10;
        end
        
        %%% folder to store results
        folderResultCur = fullfile(folderResult, [taskTestCur,'_',setTestCur,'_x',num2str(scale)]);
        if ~exist(folderResultCur,'file')
            mkdir(folderResultCur);
        end
        
        for i = 1 : length(filepaths)
            
            HR  = imread(fullfile(folderTestCur,filepaths(i).name));
            [~,imageName,ext] = fileparts(filepaths(i).name);
            HR  = modcrop(HR, crop);
            %%% label_RGB (uint8)
            label_RGB = HR;
            chanel = size(HR,3);
            %%% LR (uint8)
            LR = imresize(HR,1/scale,'bicubic');
            if chanel == 3
                %%% label (single)
                HR_ycc = single(rgb2ycbcr(im2double(HR)));
                label  = HR_ycc(:,:,1);
                %%% input (single)
                HR_bic     = imresize(im2double(LR),scale,'bicubic');
                LR_bic_ycc = rgb2ycbcr(HR_bic);
                input      = im2single(LR_bic_ycc(:,:,1));
                %%% input_RGB (uint8)
                input_RGB  = im2uint8(HR_bic);
            else
                %%% label (single)
                label  = im2single(HR);
                HR_bic = imresize(LR,scale,'bicubic');
                %%% input (single)
                input  = im2single(HR_bic);
                %%% input_RGB (uint8)
                input_RGB = HR_bic;
            end
            
            if useGPU
                input = gpuArray(input);
            end
            res = vl_simplenn(net, input,[],[],'conserveMemory',true,'mode','test');
            im = res(end).x;
            
            %%% output (single)
            output = gather(input - im);
            if chanel == 3
                %%% output_RGB (uint8)
                LR_bic_ycc(:,:,1) = double(output);
                output_RGB = im2uint8(ycbcr2rgb(LR_bic_ycc));
            else
                %%% output_RGB (uint8)
                output_RGB = im2uint8(output);
            end
            
            [PSNR_Cur,SSIM_Cur] = Cal_PSNRSSIM(label*255,output*255,ceil(scale),ceil(scale)); %%% single
            disp(['Single Image Super-Resolution     ',num2str(PSNR_Cur,'%2.2f'),'dB','    ',filepaths(i).name]);
            eval(['PSNR_SR_',setTestCur,'_x',num2str(scale),'(',num2str(i),') = PSNR_Cur;']);
            eval(['SSIM_SR_',setTestCur,'_x',num2str(scale),'(',num2str(i),') = SSIM_Cur;']);
            if showResult(2)
                imshow(cat(1,cat(2,input_RGB,output_RGB),cat(2,(output_RGB-input_RGB),label_RGB)));
                drawnow;
                title(['Single Image Super-Resolution     ',filepaths(i).name,'    ',num2str(PSNR_Cur,'%2.2f'),'dB'],'FontSize',12)
                pause(pauseTime)
                % pause()
                
                %%% save results
                imwrite(output_RGB,fullfile(folderResultCur,[imageName,'_x',num2str(scale),'.png']));
                
            end
            
        end
        disp(['Average PSNR is ',num2str(mean(eval(['PSNR_',taskTestCur,'_',setTestCur,'_x',num2str(scale)])),'%2.2f'),'dB']);
        disp(['Average SSIM is ',num2str(mean(eval(['SSIM_',taskTestCur,'_',setTestCur,'_x',num2str(scale)])),'%2.4f')]);
        
        %%% save PSNR and SSIM metrics
        save(fullfile(folderResultCur,['PSNR_',taskTestCur,'_',setTestCur,'_x',num2str(scale),'.mat']),['PSNR_',taskTestCur,'_',setTestCur,'_x',num2str(scale)])
        save(fullfile(folderResultCur,['SSIM_',taskTestCur,'_',setTestCur,'_x',num2str(scale),'.mat']),['SSIM_',taskTestCur,'_',setTestCur,'_x',num2str(scale)])
        
    end
end

%%%-------------------------------------------------------------------------------------
%%% JPEG Image Deblocking (DB)
%%%-------------------------------------------------------------------------------------

if ismember('DB',taskTest)
    taskTestCur = 'DB';
    for n_set = 1 : numel(setTest{3})
        %%% read image names
        setTestCur = cell2mat(setTest{3}(n_set));
        disp('---------------------------------------');
        disp(['----',setTestCur,'------Deblocking-----']);
        disp('---------------------------------------');
        folderTestCur = fullfile(folderTest,setTestCur);
        ext                 =  {'*.jpg','*.png','*.bmp'};
        filepaths           =  [];
        for i = 1 : length(ext)
            filepaths = cat(1,filepaths,dir(fullfile(folderTestCur, ext{i})));
        end
        
        %%% to store PSNR and SSIM results
        eval(['PSNR_',taskTestCur,'_',setTestCur,'_q',num2str(Q),' = zeros(length(filepaths),1);']);
        eval(['SSIM_',taskTestCur,'_',setTestCur,'_q',num2str(Q),' = zeros(length(filepaths),1);']);
        
        %%% to store results
        folderResultCur = fullfile(folderResult, [taskTestCur,'_',setTestCur,'_q',num2str(Q)]);
        if ~exist(folderResultCur,'file')
            mkdir(folderResultCur);
        end
        
        for i = 1 : length(filepaths)
            label  = imread(fullfile(folderTestCur,filepaths(i).name));
            [~,imageName,ext] = fileparts(filepaths(i).name);
            chanel = size(label,3);
            if chanel == 3
                %%% label (uint8)
                label = rgb2ycbcr(label);
                label = label(:,:,1);
            end
            %%% input (single)
            imwrite(label,'test.jpg','jpg','quality',Q);
            input = im2single(imread('test.jpg'));
            
            if useGPU
                input = gpuArray(input);
            end
            res = vl_simplenn(net, input,[],[],'conserveMemory',true,'mode','test');
            im = res(end).x;
            
            %%% output (single)
            output = gather(input - im);
            
            [PSNR_Cur,SSIM_Cur] = Cal_PSNRSSIM(label,im2uint8(output),0,0);
            disp(['Deblocking     ',num2str(PSNR_Cur,'%2.2f'),'dB','    ',filepaths(i).name]);
            eval(['PSNR_',taskTestCur,'_',setTestCur,'_q',num2str(Q),'(',num2str(i),') = PSNR_Cur;']);
            eval(['SSIM_',taskTestCur,'_',setTestCur,'_q',num2str(Q),'(',num2str(i),') = SSIM_Cur;']);
            
            if showResult(3)
                imshow(cat(1,cat(2,im2uint8(input),im2uint8(output)),cat(2,im2uint8(abs(input-output)*10),label)));
                drawnow;
                title(['Deblocking     ',filepaths(i).name,'    ',num2str(PSNR_Cur,'%2.2f'),'dB'],'FontSize',12)
                pause(pauseTime)
                
                %%% save results
                imwrite(output,fullfile(folderResultCur,[imageName,'_q',num2str(Q),'.png']));
                
            end
            
        end
        disp(['Average PSNR is ',num2str(mean(eval(['PSNR_',taskTestCur,'_',setTestCur,'_q',num2str(Q)])),'%2.2f'),'dB']);
        disp(['Average SSIM is ',num2str(mean(eval(['SSIM_',taskTestCur,'_',setTestCur,'_q',num2str(Q)])),'%2.4f')]);
        
        %%% save PSNR and SSIM metrics
        save(fullfile(folderResultCur,['PSNR_',taskTestCur,'_',setTestCur,'_q',num2str(Q),'.mat']),['PSNR_',taskTestCur,'_',setTestCur,'_q',num2str(Q)])
        save(fullfile(folderResultCur,['SSIM_',taskTestCur,'_',setTestCur,'_q',num2str(Q),'.mat']),['SSIM_',taskTestCur,'_',setTestCur,'_q',num2str(Q)])
        
    end
end


























