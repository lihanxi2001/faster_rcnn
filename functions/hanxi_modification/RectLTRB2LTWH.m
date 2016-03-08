% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Thursday 31/12/2015 12:18.
% Last Revision: Thursday 31/12/2015 12:18.

function [ rectsLTWH ] = RectLTRB2LTWH( rectsLTRB )
%rects (l, t, r, b) to (l, t, w, h)

rectsLTWH = [rectsLTRB(:, 1), ...
    rectsLTRB(:, 2), rectsLTRB(:, 3)-rectsLTRB(:,1)+1, rectsLTRB(:,4)-rectsLTRB(:, 2)+1];

end


