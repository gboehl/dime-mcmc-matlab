% marginal pdf of test func

function mpdf = dime_test_func_marginal_pdf(x, cov_scale, distance, weight)

    pdf_left = normpdf(x + distance, 0, sqrt(cov_scale));
    pdf_mid = normpdf(x, 0, sqrt(cov_scale));
    pdf_right = normpdf(x - distance, 0, sqrt(cov_scale));

    mpdf = weight(1)*pdf_left + weight(2)*pdf_mid + (1-weight(1)-weight(2))*pdf_right;

end
