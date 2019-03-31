function dn = backprop(da,n,a,param)
%TANSIG.BACKPROP Backpropagate derivatives from outputs to inputs

% Copyright 2012-2015 The MathWorks, Inc.
  nettansigR.dd_offset = 0;
  dn = bsxfun(@times,da,1-(a.*a)+  nettansigR.dd_offset);
end
