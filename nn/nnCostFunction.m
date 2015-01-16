function [J grad] = nnCostFunction(nn_params, nn_lsizes, X, y, lambda)
a = X; %contains horizontal vectors of neuron activation values;
indices_a = []; %contains start-end indices for neurons of different layers (for grouping neuron activation values by layers)
Jvec_params = []; %used to regularize Cost Function
gradvec_params = []; %used to regularize gradients
a_pos = 0;
params_pos = 0;
for i=1:size(nn_lsizes,2)-1;
    ThetaRows = nn_lsizes(i+1);
    ThetaCols = nn_lsizes(i)+1;
    Theta = reshape(nn_params(params_pos+1 : params_pos+ThetaRows*ThetaCols), ThetaRows, ThetaCols);
    
    a = [a  sigmoid([ones(size(a,1),1) a(:,a_pos+1:a_pos+nn_lsizes(i))] * Theta')];

    indices_a = [ indices_a [a_pos+1;a_pos+nn_lsizes(i)]];

    Jvec_params = [Jvec_params; Theta(:,2:end)(:)];
    Theta(:,1)=0;
    gradvec_params = [gradvec_params; Theta(:)];

    params_pos += ThetaRows*ThetaCols;
    a_pos += nn_lsizes(i);
end
indices_a = [indices_a [a_pos+1;a_pos+nn_lsizes(end)]];

% COST
Jvec_h = a(:,end-nn_lsizes(end)+1:end)(:);
Jvec_y = y(:);
%% Eliminates cases when hypothesis is equal to y. Useful to evade log(0) calculation
id_nonEqual = find(Jvec_h != Jvec_y);
Jvec_h = Jvec_h(id_nonEqual);
Jvec_y = Jvec_y(id_nonEqual);
%%
J = -1/size(a,1) .* (Jvec_y'*log(Jvec_h) + (1-Jvec_y)'*log(1-Jvec_h));
J += lambda/(2*size(a,1)) * sum(sum(Jvec_params.*Jvec_params));

% GRADIENTS
grad = (  1/size(a,1)  *  (a(:,end-nn_lsizes(end)+1:end) - y)'  *  [ones(size(a,1),1) a(:,indices_a(1,size(nn_lsizes,2)-1):indices_a(2,size(nn_lsizes,2)-1))]  )(:);
delta = a(:,end-nn_lsizes(end)+1:end) - y;
for i=fliplr(3:size(nn_lsizes,2))
    ThetaRows = nn_lsizes(i);
    ThetaCols = nn_lsizes(i-1)+1;
    params_pos -= ThetaRows*ThetaCols;
    Theta = reshape(nn_params(params_pos+1 : params_pos+ThetaRows*ThetaCols), ThetaRows, ThetaCols);
    delta = delta * Theta(:,2:end) .* sigmoidGrad(a(:,indices_a(1,i-1):indices_a(2,i-1)));
    grad = [( 1/size(a,1) * delta' * [ones(size(a,1),1) a(:,indices_a(1,i-2):indices_a(2,i-2))] )(:); grad];       
end
grad += lambda/size(a,1) .* gradvec_params;
endfunction

