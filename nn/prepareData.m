function [X, y, mu, s, X_test, y_test, X_cv, y_cv] = prepareData(data, fractionTest=0, fractionCV=0)
X=y=X_cv=y=cv=X_test=y_test=[];

% Scales (and normalizes) features
features = data(:,1:end-1);
mu = mean(features, 1);
s = std(features, [], 1);
features -= (ones(size(data,1),1) * mu);
features ./= (ones(size(data,1),1) * s);
data(:,1:end-1) = features;

% Randomly shuffles dataset
data = data(randperm(size(data,1)), :);

% Split dataset examples
if (fractionTest)
    size_test = round(size(data,1) * fractionTest);
    data_test = data((1:size_test), :);
    X_test = data_test(:,1:end-1);
    y_test = data_test(:,end);

    data = data(size_test+1:end, :);
end

if (fractionCV)
    size_cv = round(size(data,1) * (1-fractionTest) * fractionCV);
    data_cv = data((1:size_test), :);
    X_cv = data_cv(:,1:end-1);
    y_cv = data_cv(:,end);    

    data = data(size_test+1:end, :);
end

data_train = data;
X = data_train(:,1:end-1);
y = data_train(:,end);

endfunction

