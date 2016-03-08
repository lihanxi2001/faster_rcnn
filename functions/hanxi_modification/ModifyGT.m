% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Sunday 17/01/2016 11:15.
% Last Revision: Sunday 17/01/2016 11:15.

function roidb = ModifyGT(roidb)

    for i = 1 : length(roidb.rois)
        roidb.rois(i).gt = roidb.rois(i).gt(:);
    end

end



