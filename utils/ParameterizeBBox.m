% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Thursday 25/02/2016 12:02.
% Last Revision: Thursday 25/02/2016 12:02.

function [boxes_param, boxes] = ParameterizeBBox(boxes, boxes_param)

    if ~isempty(boxes)
        box_x = 0.5 * (boxes(1, :) + boxes(3, :));
        box_y = 0.5 * (boxes(2, :) + boxes(4, :));
        box_h = boxes(4, :) - boxes(2, :);
        box_w = boxes(3, :) - boxes(1, :);
        box_log_ar = log(box_h ./ box_w + 1e-5);
        box_log_s = log(box_h / 32);
        boxes_param = [box_x(:)'; box_y(:)'; box_log_ar(:)'; box_log_s(:)'];
    end

    if ~isempty(boxes_param)
        box_x = boxes_param(1, :);
        box_y = boxes_param(2, :);
        box_log_ar = boxes_param(3, :);
        box_log_s = boxes_param(4, :);
        box_h = exp(box_log_s) * 32;
        box_w = box_h ./ exp(box_log_ar);
        box_l = box_x - box_w * 0.5;
        box_r = box_x + box_w * 0.5;
        box_t = box_y - box_h * 0.5;
        box_b = box_y + box_h * 0.5;
        boxes = [box_l(:)'; box_r(:)'; box_t(:)'; box_b(:)'];
    end

end



