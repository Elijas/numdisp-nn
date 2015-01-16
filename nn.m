function nn
addpath('nn')

X = (1:16)';
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

nn_lambda = .0001;
nn_lsizes = [size(X,2) 7];
nn_options = optimset('MaxIter', 600);
nn_params = fmincg(@(p) nnCostFunction(p, nn_lsizes, X, y, nn_lambda), nnInitParams(nn_lsizes), nn_options);

%{
params_pos=0;
for i=1:size(nn_lsizes,2)-1;
    ThetaRows = nn_lsizes(i+1);
    ThetaCols = nn_lsizes(i)+1;
    Theta = reshape(nn_params(params_pos+1 : params_pos+ThetaRows*ThetaCols), ThetaRows, ThetaCols)
end
%}

for i=X'
    printf('input: %d expected output:\n',i)
    disp(y(i,:))
    print_output(y(i,:))
    printf('\nNN output:\n',i)
    disp(nnFeedForward(nn_params, nn_lsizes, i))
    print_output(round(nnFeedForward(nn_params, nn_lsizes, i)))
    printf('\n\n\n');
end

accuracy = mean(  (  y==round(nnFeedForward(nn_params, nn_lsizes, X))  )(:)  );
printf('layers: '), printf(' %d', nn_lsizes), printf('\t| accuracy: %10.2f%%\n', accuracy*100);

endfunction

