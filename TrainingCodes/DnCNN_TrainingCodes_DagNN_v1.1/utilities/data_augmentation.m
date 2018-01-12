function image = data_augmentation(image, mode)

if mode == 1
    return;
end

if mode == 2 % flipped
    image = flipud(image);
    return;
end

if mode == 3 % rotation 90
    image = rot90(image,1);
    return;
end

if mode == 4 % rotation 90 & flipped
    image = rot90(image,1);
    image = flipud(image);
    return;
end

if mode == 5 % rotation 180
    image = rot90(image,2);
    return;
end

if mode == 6 % rotation 180 & flipped
    image = rot90(image,2);
    image = flipud(image);
    return;
end

if mode == 7 % rotation 270
    image = rot90(image,3);
    return;
end

if mode == 8 % rotation 270 & flipped
    image = rot90(image,3);
    image = flipud(image);
    return;
end

















