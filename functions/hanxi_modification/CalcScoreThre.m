% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Sunday 06/03/2016 00:28.
% Last Revision: Sunday 06/03/2016 00:28.

function thre = CalcScoreThre(r, label, score)

    [sort_score, sort_idx] = sort(score, 'descend');
    sort_label = label(sort_idx);
    recall_rate = cumsum(sort_label > 0) / sum(label > 0);
    thre_idx = find(recall_rate > r, 1, 'first');
    thre = sort_score(thre_idx);

end



