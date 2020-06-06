## Copyright (C) 2020 Tushar Upadhyay

function display1 (X)
function h = subplottight(n,m,i)
    [c,r] = ind2sub([m n], i);
    ax = subplot('Position', [(c-1)/m, 1-(r)/n, 1/m, 1/n])
    if(nargout > 0)
      h = ax;
    end
    endfunction
for i=1:10
  t = reshape(X(i,:),[20,20]);
  
  subplottight(10,10,i,[0.05,0.05])
  imshow(t)

endfor
endfunction
