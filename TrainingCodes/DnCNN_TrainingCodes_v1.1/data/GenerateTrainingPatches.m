
%%% Generate the training data.

clear;close all;

addpath('utilities');

batchSize      = 128;        %%% batch size
dataName      = 'TrainingPatches';
folder        = 'Train400';

patchsize     = 40;
stride        = 10;
step          = 0;

count   = 0;

ext               =  {'*.jpg','*.png','*.bmp','*.jpeg'};
filepaths           =  [];

for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(folder, ext{i})));
end

%% count the number of extracted patches
scales  = [1 0.9 0.8 0.7];
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name)); % uint8
    if size(image,3)==3
        image = rgb2gray(image);
    end
    %[~, name, exte] = fileparts(filepaths(i).name);
    if mod(i,100)==0
        disp([i,length(filepaths)]);
    end
    for s = 1:4
        image = imresize(image,scales(s),'bicubic');
        [hei,wid,~] = size(image);
        for x = 1+step : stride : (hei-patchsize+1)
            for y = 1+step :stride : (wid-patchsize+1)
                count = count+1;
            end
        end
    end
end

numPatches = ceil(count/batchSize)*batchSize;

disp([numPatches,batchSize,numPatches/batchSize]);

%pause;

inputs  = zeros(patchsize, patchsize, 1, numPatches,'single'); % this is fast
count   = 0;
tic;
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name)); % uint8
    %[~, name, exte] = fileparts(filepaths(i).name);
    if size(image,3)==3
        image = rgb2gray(image);
    end
    if mod(i,100)==0
        disp([i,length(filepaths)]);
    end
    %     end
    for s = 1:4
        image = imresize(image,scales(s),'bicubic');
        for j = 1:1
            image_aug = data_augmentation(image, j);  % augment data
            im_label  = im2single(image_aug); % single
            [hei,wid,~] = size(im_label);
            
            for x = 1+step : stride : (hei-patchsize+1)
                for y = 1+step :stride : (wid-patchsize+1)
                    count       = count+1;
                    inputs(:, :, :, count)   = im_label(x : x+patchsize-1, y : y+patchsize-1,:);
                end
            end
        end
    end
end
toc;
set    = uint8(ones(1,size(inputs,4)));

disp('-------Datasize-------')
disp([size(inputs,4),batchSize,size(inputs,4)/batchSize]);

if ~exist(dataName,'file')
    mkdir(dataName);
end

%%% save data
save(fullfile(dataName,['imdb_',num2str(patchsize),'_',num2str(batchSize)]), 'inputs','set','-v7.3')

