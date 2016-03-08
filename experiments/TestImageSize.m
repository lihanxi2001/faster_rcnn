% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Friday 11/12/2015 04:44.
% Last Revision: Friday 11/12/2015 04:44.

function TestImageSize(imdb)

% for i_db = 1 : length(imdb)
%     imdb_new = imdb{i_db};
%     for i = 1 : length(imdb_new.image_ids)
%         im = imread([imdb_new.image_dir,'/',imdb_new.image_ids{i},'.jpg']);
%         assert(all(size(im, 1) == imdb_new.sizes(i, 1)));
%         assert(all(size(im, 2) == imdb_new.sizes(i, 2)));
%         fprintf('%d == %d, %d == %d.\n', size(im, 1), imdb_new.sizes(i, 1), size(im, 2), imdb_new.sizes(i, 2));
%     end
% end

% for i_db = 1 : length(imdb)
%     imdb_new = imdb{i_db};
    for i = 1 : length(imdb)
        im = imread([imdb(i).image_dir,'/',imdb(i).image_id,'.jpg']);
        assert(all(size(im, 1) == imdb(i).sizes(i, 1)));
        assert(all(size(im, 2) == imdb(i).sizes(i, 2)));
        fprintf('%d == %d, %d == %d.\n', size(im, 1), imdb(i).sizes(1), size(im, 2), imdb(i).sizes(2));
    end
% end

return



