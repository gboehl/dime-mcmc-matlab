function logpdf = create_dime_test_func(ndim, weight, distance, scale)
% create a test function

covm = eye(ndim)*scale;

meanm = zeros(ndim,1);
meanm(1) = distance;

w1 = weight(1);
w2 = weight(2);
w3 = 1-weight(1)-weight(2);

logpdf = @(p) log(w1*mvnpdf(p' + meanm', zeros(ndim,1), covm) ...
    + w2*mvnpdf(p', zeros(ndim,1), covm) ...
    + w3*mvnpdf(p' - meanm', zeros(ndim,1), covm));
