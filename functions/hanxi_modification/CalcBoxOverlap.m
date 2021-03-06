% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Saturday 25/01/2014 17:33.
% Last Revision: Saturday 25/01/2014 17:33.
%Bit

function dist = CalcBoxOverlap(A, B, dist_type)

% GETBOXOVERLAP
%   A and B have a box for each column, in the format [xmin ymin xmax
%   ymax]. The resulting matrix dist has A's boxes along the rows
%   and B's boxes along the columns.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

m = size(A,2) ;
n = size(B,2) ;
O = [] ;

if m==0 || n==0, dist = zeros(m,n) ; return ; end

om = ones(1,m) ;
on = ones(1,n) ;

% find length Ox of the overlap range [x1, x2] along x
% x1 cannot be smaller than A.xmin B.xmin
% x2 cannot be larger  than A.xmax B.xmax
% Ox is x2 - x1 or 0

x1 = max(A(1*on,:)', B(1*om,:)) ;
x2 = min(A(3*on,:)', B(3*om,:)) ;
Ox = max(x2 - x1, 0) ;

y1 = max(A(2*on,:)', B(2*om,:)) ;
y2 = min(A(4*on,:)', B(4*om,:)) ;
Oy = max(y2 - y1, 0) ;

% are of the intersection
areaInt = Ox .* Oy ;

% area of the union is sum of areas - inersection
areaA = prod(A(3:4,:) - A(1:2,:)) ;
areaB = prod(B(3:4,:) - B(1:2,:)) ;

% final distance matrix
switch dist_type
case 'int_uni'
    dist = areaInt ./ (areaA(on,:)' + areaB(om,:) - areaInt) ;
case 'int_min'
    dist = areaInt ./ min(areaA(on,:)', areaB(om,:));
% case 'int_area'
%     dist = areaInt ./ mean(areaA(on,:)' + areaB(om,:) - areaInt);
otherwise
end

return;
