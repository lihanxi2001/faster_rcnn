function [boxes, prior_scores, deep_fea] = PriorDetection(img_ids, prior_res, fc6)
    aboxes = prior_res.aboxes_all{1}(img_ids);
    aboxes = aboxes{1};
    boxes = aboxes(:,1:4);
    prior_scores = aboxes(:,5);
    
    if fc6
        deep_fea = prior_res.deep_fea6_all{img_ids};
    else
        deep_fea = prior_res.deep_fea7_all(img_ids);
    end
end