function [net1] = vl_simplenn_mergebnorm(net)

%% merge bnorm parameters into adjacent Conv layer

for i = 1:numel(net.layers)
    if strcmp(net.layers{i}.type, 'conv')
        net.layers{i}.weightDecay(2) = 1;
    end
end

for i = 1:numel(net.layers)
    if strcmp(net.layers{i}.type, 'bnorm')
        ws = net.layers{i}.weights{1};
        bs = net.layers{i}.weights{2};
        mu_sigmas = net.layers{i}.weights{3};
        for j = 1:numel(ws)
            net.layers{i-1}.weights{1}(:,:,:,j) =single(double(net.layers{i-1}.weights{1}(:,:,:,j))*double(ws(j))/(double(mu_sigmas(j,2))));
            net.layers{i-1}.weights{2}(j) =single(double(bs(j)) - double(ws(j))*double(mu_sigmas(j,1))/(double(mu_sigmas(j,2))));
        end
        net.layers{i-1}.learningRate(2) = 1;
    end
end

net1 = net;
net1.layers = {};
net1 = rmfield(net1,'meta');
for i = 1:numel(net.layers)
    if ~strcmp(net.layers{i}.type, 'bnorm')
        net1.layers{end+1} = net.layers{i};
    end
end

net1.layers = net1.layers(1:end-1);


end
