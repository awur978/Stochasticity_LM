function a = apply(n,param)
%TANSIG.APPLY Apply transfer function to inputs

% Copyright 2012-2015 The MathWorks, Inc.

       % R = net.activation_noise;
	   R = 0.01;
          rn = R * (rand(size(n))*2-1) .* n/1; 
        a = 2 ./ (1 + exp(-n)) - 1 + rn ;
end


