function a = tansigR(n,varargin)
%TANSIG Symmetric sigmoid transfer function.
%
if ischar(n)
  a = nnet7.transfer_fcn(mfilename,n,varargin{:});
  return
end

% Apply
a = tansigR.apply(n);

