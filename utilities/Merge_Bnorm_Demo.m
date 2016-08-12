




load('sigma=25_Bnorm.mat');

[net] = vl_simplenn_mergebnorm(net);

save sigma=25 net;


