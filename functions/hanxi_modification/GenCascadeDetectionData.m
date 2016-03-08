% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Sunday 06/03/2016 01:51.
% Last Revision: Sunday 06/03/2016 01:51.

function [prior_scores, deep_fea, label] = ...
        GenCascadeDetectionData(scores, boxes, ...
                        deep_fea, gt_boxes, pos_thre, neg_thre, use_gpu)

    % Overlap between boxes.
    box_overlap = CalcBoxOverlap(gt_boxes', boxes', 'int_uni');
    box_overlap = max(box_overlap, [], 1);

    % Positive boxes and scores.
    pos_idx = box_overlap > pos_thre;
    fg_boxes = boxes(pos_idx, :);
    fg_scores = scores(pos_idx);
    fg_fea = deep_fea(:, pos_idx);
    nms_pick = nms([fg_boxes, fg_scores(:)], 0.6, use_gpu);
%     fg_boxes = fg_boxes(nms_pick, :);
    fg_scores = fg_scores(nms_pick);
    n_pos = numel(fg_scores);

    % Negative boxes and scores.
    neg_idx = box_overlap < neg_thre;
    bg_boxes = boxes(neg_idx, :);
    bg_scores = scores(neg_idx);
    bg_fea = deep_fea(:, neg_idx);
    nms_pick = nms([bg_boxes, bg_scores(:)], 0.5, use_gpu);
%     bg_boxes = bg_boxes(nms_pick, :);
    bg_scores = bg_scores(nms_pick);
    if numel(bg_scores) > 2 * n_pos
        rand_idx = randperm(length(bg_scores));
%         bg_boxes = bg_boxes(rand_idx(1 : 2 * n_pos), :);
        bg_fea = bg_fea(:, rand_idx(1 : 2 * n_pos));
        bg_scores = bg_scores(rand_idx(1 : 2 * n_pos));
    end
    n_neg = numel(bg_scores);

    prior_scores = [fg_scores; bg_scores];
    deep_fea = [fg_fea, bg_fea];
    label = [ones(1, n_pos), -ones(1, n_neg)];

end



