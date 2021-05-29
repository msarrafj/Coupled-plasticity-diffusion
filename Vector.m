function [vecA] = Vector(A)
% Input variables
% A: given matrix (of size m x n)
% Output variables
% vecA: contains A in vector form stacked one column after another
mA = size(A,1);
nA = size(A,2);
vecA = zeros(mA*nA,1);
for col = 1:nA
    for row = 1:mA
        vecA((col-1)*mA+row) = A(row,col);
    end
end
