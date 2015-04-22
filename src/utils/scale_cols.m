function y = scale_cols(x, s)
y = x.*repmat(s(:).', size(x,1), 1);
