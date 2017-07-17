setTestCur = 'Set14';
ext                 =  {'*.jpg','*.png','*.bmp'};
filepaths           =  [];
for i = 1 : length(ext)
    filepaths = cat(1,filepaths,dir(fullfile(setTestCur, ext{i})));
end
%%% folder to store results
folderResultCur = [setTestCur,'G'];
if ~exist(folderResultCur,'file')
    mkdir(folderResultCur);
end
for i = 1 : length(filepaths)

    color  = imread(fullfile(setTestCur,filepaths(i).name));
    [~,imageName,ext] = fileparts(filepaths(i).name);
    channel = size(color,3);

    if channel == 3
        %%% label (single)
        HR_ycc = single(rgb2ycbcr(im2double(color)));
        label  = HR_ycc(:,:,1);
        output_RGB = im2uint8(label);
    else
        output_RGB = im2uint8(color);
    end
    imwrite(output_RGB,fullfile(folderResultCur,[imageName,'.bmp']));
end