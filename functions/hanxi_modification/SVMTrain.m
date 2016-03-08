% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Sunday 06/03/2016 00:38.
% Last Revision: Sunday 06/03/2016 00:38.

function svm_model = SVMTrain(label, fea, n_cross)

    % cross validation
    c = 2.^(-4 : 4);
    accu = zeros(size(c));
    fea = sparse(fea);
%     label = sparse(label);
    for i_c = 1 : numel(c)
    accu(i_c) = libsvmtrain(ones(size(label)), label, fea, ...
            ['-q -c ', num2str(c(i_c)), ' -v ', num2str(n_cross)]);
    end

    % pick a parameter
    [~, max_idx] = max(accu);

    % training
    svm_model = libsvmtrain(ones(size(label)), ...
                label, fea, ['-c ', num2str(c(max_idx))]);

end



