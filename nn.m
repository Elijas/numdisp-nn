function nn
addpath('nn')

X = eye(16); %TODO: Try without binary expansion
y = [ ...
0,0,0,1,0,0,1;% 1
1,0,1,1,1,1,0;% 2
1,0,1,1,0,1,1;
0,1,1,1,0,0,1;
1,1,1,0,0,1,1;
1,1,1,0,1,1,1;
1,1,0,1,0,0,1;
1,1,1,1,1,1,1;
1,1,1,1,0,1,1;% 9
1,1,0,1,1,1,1;% 0
1,1,1,1,1,0,1;% A
0,1,1,0,1,1,1;% B
1,1,0,0,1,1,0;
0,0,1,1,1,1,1;
1,1,1,0,1,1,0;
1,1,1,0,1,0,0;% F
];

nn_lambda = .001;
nn_lsizes = [size(X,2) 7];
nn_options = optimset('MaxIter', 600);
nn_params = fmincg(@(p) nnCostFunction(p, nn_lsizes, X, y, nn_lambda), nnInitParams(nn_lsizes), nn_options)
    
accuracy = mean(  (  y==round(nnFeedForward(nn_params, nn_lsizes, X))  )(:)  );
printf(' %d', nn_lsizes), printf('\t| %10.2f%%\n', accuracy*100);

for i=X
    printf('input: %d  output:\n',find(i))
    print_output(round(nnFeedForward(nn_params, nn_lsizes, i')))
end
endfunction

