% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Sunday 06/03/2016 00:20.
% Last Revision: Sunday 06/03/2016 00:49.

function [post_score, predict_label] = ...
        CascadeDetectionTest(prior_score, deep_fea, detect_model)

    pass_idx = prior_score > detect_model.score_thre;
    label = rand(size(deep_fea, 2), 1);
    deep_fea = deep_fea(:, pass_idx);

    switch detect_model.method
    case 'svm'
        [predict_label, ~, post_score] = ...
            svmpredict(label, sparse(double(deep_fea')), detect_model.model);
    case 'rf'
        error('I do not know what is rf.');
    otherwise
        error('What the hell is this method?');
    end

end



