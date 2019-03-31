function a = apply(n,param)
%ELLIOTSIG.APPLY Apply transfer function to inputs

% Copyright 2012-2015 The MathWorks, Inc.
	R = 0.1;
    rn = R * (rand(size(n))*2-1) .* n/1; 
a = n ./ (1 + abs(n))  + rn ;
end


