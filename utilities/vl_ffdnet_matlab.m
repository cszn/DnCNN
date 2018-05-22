function res = vl_ffdnet_matlab(net, input)

%% If you did not install the matconvnet package, you can use this for testing.

global sigmas;
n = numel(net.layers);
res = struct('x', cell(1,n+1));
res(1).x = input;

for i = 1 : n
    l = net.layers{i};
    switch l.type
        
        case 'conv'
            disp(['Processing ... ',int2str(i),'/',int2str(n)]);
            for noutmaps = 1 : size(l.weights{1},4)
                z = zeros(size(res(i).x,1),size(res(i).x,2),'single');
                for ninmaps = 1 : size(res(i).x,3)
                    z = z + convn(res(i).x(:,:,ninmaps), rot90(l.weights{1}(:,:,ninmaps,noutmaps),2),'same'); % 180 degree rotation for kernel
                end
                res(i+1).x(:,:,noutmaps) = z + l.weights{2}(noutmaps);
            end
            
        case 'relu'
            res(i+1).x = max(res(i).x,0);
            
        case 'concat'
            if size(sigmas,1)~=size(res(i).x,1)
                sigmaMap   = bsxfun(@times,ones(size(res(i).x,1),size(res(i).x,2),1,size(res(i).x,4),'single'),permute(sigmas,[3 4 1 2]));
                res(i+1).x = cat(3,res(i).x,sigmaMap);
            else
                res(i+1).x = cat(3,res(i).x,sigmaMap);
            end
            
        case 'SubP'
            res(i+1).x = vl_nnSubP(res(i).x, [],'scale',l.scale);
            
    end
    res(i).x = [];
end

end
