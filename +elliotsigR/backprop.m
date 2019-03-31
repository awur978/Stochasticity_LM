function dn = backprop(da,n,a,param)
%ELLIOTSIG.BACKPROP Backpropagate derivatives from outputs to inputs

% Copyright 2012-2015 The MathWorks, Inc.
  nettansigR.dd_offset = 0;
  d = 1 ./ ((1+abs(n)).^2) +  nettansigR.dd_offset;
  dn = bsxfun(@times,da,d);
end

