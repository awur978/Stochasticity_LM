function da = forwardprop(dn,n,a,param)
%ELLIOTSIG.FORWARDPROP Forward propagate derivatives from input to output.

% Copyright 2012-2015 The MathWorks, Inc.
  nettansigR.dd_offset = 0;
  d = 1 ./ ((1+abs(n)).^2) + nettansigR.dd_offset;
  da = bsxfun(@times,dn,d);
end
