% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Sunday 06/03/2016 00:20.
% Last Revision: Sunday 06/03/2016 12:55.

function detect_model = CascadeDetectionTrain(prior_scores, ...
                                        label, deep_fea, method)

    r = 0.95; % Pass r positive samples to the next detector.
    score_thre = CalcScoreThre(r, label, prior_scores);
    pass_idx = prior_scores > score_thre;
    label = label(pass_idx);
    deep_fea = deep_fea(:, pass_idx);

    switch method
    case 'svm'
        n_cross = 5;
        svm_model = SVMTrain(double(label(:)), ...
                    double(deep_fea'), n_cross);
    case 'rf'
        error('I do not know what is rf.');
    otherwise
        error('What the hell is this method?');
    end
    detect_model.model = svm_model;
    detect_model.prior_score_thre = score_thre;
    detect_model.method = method;

end



