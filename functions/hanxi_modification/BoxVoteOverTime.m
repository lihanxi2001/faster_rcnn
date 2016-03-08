% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Thursday 25/02/2016 12:28.
% Last Revision: Thursday 25/02/2016 12:28.

function [smooth_score] = SmoothBoxOverTime(boxes, scores, ...
                    prev_boxes, prev_scores, f_diff, temporal_s, T)

    if nargin == 5
        temporal_s = 0.75;
        T = 4;
    end

    valid_idx = f_diff <= T;
    prev_boxes = prev_boxes(:, valid_idx);
    prev_scores = prev_scores(valid_idx);
    f_diff = f_diff(valid_idx);

    boxes_param = ParameterizeBBox(boxes, []);
    prev_boxes_param = ParameterizeBBox(prev_boxes, []);

    f_diff = [zeros(1, numel(scores)), f_diff(:)'];

    temporal_factor = exp(-temporal_s .* (f_diff(:)' - 0.5));

    U = bsxfun(@times, spatial_factor, ...
            temporal_factor .* prev_scores(:)');
    V = bsxfun(@times, U, scores(:));
    sort_V = sort(V, 2, 'descend');
    sort_V = sort_V(:, 1 : 20);
    smooth_score = scores(:) + mean(sort_V, 2);

end


function FindBoxesRange(boxes_param, scores)

    inter_xy = 4;
    inter_ar = 0.05;
    inter_s = 0.05;

    max_score = max(socres);
    large_idx = scores > 0.5 * max_score;
    boxes_param = boxes_param(:, large_idx);
    max_x = max(boxes_param(1, :));
    min_x = min(boxes_param(1, :));
    max_y = max(boxes_param(2, :));
    min_y = min(boxes_param(2, :));
    max_ar = max(boxes_param(3, :));
    min_ar = min(boxes_param(3, :));
    max_s = max(boxes_param(4, :));
    min_s = min(boxes_param(4, :));

    T_x = ceil((max_x - min_x) / inter_xy) + 1;
    T_y = ceil((max_y - min_y) / inter_xy) + 1;
    T_ar = ceil((max_ar - min_ar) / inter_ar) + 1;
    T_s = ceil((max_s - min_s) / inter_s) + 1;

    x = round((boxes_param(1, :) - min_x) / inter_xy) + 1;
    y = round((boxes_param(2, :) - min_y) / inter_xy) + 1;
    ar = round((boxes_param(3, :) - min_ar) / inter_ar) + 1;
    s = round((boxes_param(4, :) - min_s) / inter_s) + 1;

    A = accumarray([x(:)'; y(:)'; ar(:)'; s(:)'], ...
                    scores(:), [T_x, T_y, T_ar, T_s]);

end

function


