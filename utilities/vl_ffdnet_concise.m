function res = vl_ffdnet_concise(net, x)

global sigmas;
n = numel(net.layers);
res = struct('x', cell(1,n+1));
res(1).x = x ;
cudnn = {'CuDNN'} ;
%cudnn = {'NoCuDNN'} ;

for i=1:n
    l = net.layers{i} ;
    switch l.type
        case 'conv'
            res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                'pad', l.pad, ...
                'stride', l.stride, ...
                'dilate', l.dilate, ...
                l.opts{:}, ...
                cudnn{:}) ;
            
        case 'concat'
            if size(sigmas,1)~=size(res(i).x,1)
                sigmaMap   = bsxfun(@times,ones(size(res(i).x,1),size(res(i).x,2),1,size(res(i).x,4),'single'),permute(sigmas,[3 4 1 2]));
                res(i+1).x = cat(3,res(i).x,sigmaMap);
            else
                res(i+1).x = cat(3,res(i).x,sigmaMap);
            end

        case 'SubP'
            res(i+1).x = vl_nnSubP(res(i).x, [],'scale',l.scale);
  
        case 'relu'
            res(i+1).x = max(res(i).x,0) ;
    end
        res(i).x = [] ;
end


