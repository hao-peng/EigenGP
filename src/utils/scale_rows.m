function y = scale_rows(x,s)
y = repmat(s(:), 1, size(x,2)).*x;
