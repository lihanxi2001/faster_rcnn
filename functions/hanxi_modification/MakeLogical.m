% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Tuesday 19/01/2016 02:50.
% Last Revision: Tuesday 19/01/2016 02:50.

function rois = MakeLogical(rois)

%     rois_new = rois;
    for i = 1 : length(rois)
        rois(i).gt = logical(rois(i).gt);
    end
%     arrayfun(@(x, y) y.gt = logical(x.gt), rois, rois_new);

end



