## Copyright (C) 2020 Tushar Upadhyay


function g = sigrad (z)
g = zeros(size(z));

g = sigmoid(z).*(1-sigmoid(z)) ;
endfunction
