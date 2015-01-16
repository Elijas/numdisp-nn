%Initializes parameter vector for a NN with random weights
function nn_params = nnInitParams(nn_lsizes)
nn_params=[];
for i=1:size(nn_lsizes,2)-1
    param_count = nn_lsizes(i+1) * (nn_lsizes(i)+1);
    epsilon = 4*sqrt( 6 / ((nn_lsizes(i)+1)+(nn_lsizes(i+1))) ); %src: http://deeplearning.net/tutorial/mlp.html
    
    nn_params = [nn_params; (rand(param_count,1)*2-1)*epsilon];
end
endfunction

