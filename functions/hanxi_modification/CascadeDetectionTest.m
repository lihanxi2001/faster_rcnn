% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Sunday 06/03/2016 00:20.
% Last Revision: Sunday 06/03/2016 00:49.

function [post_scores, predict_label] = ...
        CascadeDetectionTest(prior_scores, deep_fea, detect_model)

    post_scores = zeros(size(prior_scores));
    predict_label = zeros(size(prior_scores));
    pass_idx = prior_scores > detect_model.prior_score_thre;
    deep_fea = deep_fea(:, pass_idx);
    label = rand(size(deep_fea, 2), 1);
    
    switch detect_model.method
    case 'svm'
        [predict_label(pass_idx), ~, post_scores(pass_idx)] = ...
            libsvmpredict(label, sparse(double(deep_fea')), detect_model.model, '-q');
    case 'rf'
        error('I do not know what is rf.');
    otherwise
        error('What the hell is this method?');
    end

end



