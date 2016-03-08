function nms_pick = NMS(boxes, prior_scores, nms_thre)
%     boxes_num = size(boxes,1);
%     keep_num = round(boxes_num * nms_thre);
%     [~, nms_pick] = sort(prior_scores, 'descend');
%     nms_pick(keep_num + 1:end) = [];
    nms_pick = nms([boxes prior_scores], nms_thre);
end