% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Monday 22/02/2016 11:10.
% Last Revision: Thursday 25/02/2016 11:58.

function [smooth_score] = SmoothBoxOverTime(boxes, scores, ...
                    prev_boxes, prev_scores, f_diff, spatial_s, temporal_s, T)

    if nargin == 5
        spatial_s = 3;
        temporal_s = 0.75;
        T = 4;
    end

    valid_idx = f_diff <= T;
    prev_boxes = prev_boxes(:, valid_idx);
    prev_scores = prev_scores(valid_idx);
    f_diff = f_diff(valid_idx);

    spatial_dist = CalcBoxOverlap(boxes, prev_boxes, 'int_uni');
    spatial_factor = exp(-spatial_s .* (1 - spatial_dist));
    temporal_factor = exp(-temporal_s .* (f_diff(:)' - 0.5));

    U = bsxfun(@times, spatial_factor, ...
            temporal_factor .* prev_scores(:)');
    V = bsxfun(@times, U, scores(:));
    sort_V = sort(V, 2, 'descend');
    sort_V = sort_V(:, 1 : 20);
    smooth_score = scores(:) + mean(sort_V, 2);
%     smooth_boxes =

end



