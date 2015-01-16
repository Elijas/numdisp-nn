function [output_layer] = nnFeedForward(nn_params, nn_lsizes, X)
a = X;
indices_a = [];
a_pos = 0;
params_pos = 0;
for i=1:size(nn_lsizes,2)-1;
    ThetaRows = nn_lsizes(i+1);
    ThetaCols = nn_lsizes(i)+1;
    Theta = reshape(nn_params(params_pos+1 : params_pos+ThetaRows*ThetaCols), ThetaRows, ThetaCols);
    
    a = [a  sigmoid([ones(size(a,1),1) a(:,a_pos+1:a_pos+nn_lsizes(i))] * Theta')];
    
    indices_a = [indices_a [a_pos+1;a_pos+nn_lsizes(i)]];
    
    params_pos += ThetaRows*ThetaCols;
    a_pos += nn_lsizes(i);
end
indices_a = [indices_a [a_pos+1;a_pos+nn_lsizes(end)]];

output_layer = a(:,end-nn_lsizes(end)+1:end);
endfunction

