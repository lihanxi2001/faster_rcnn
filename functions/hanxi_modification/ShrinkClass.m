% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Thursday 10/12/2015 16:33.
% Last Revision: Friday 11/12/2015 02:46.

function [imdb_out, roidb_out] = ShrinkClass(imdb_in, roidb_in, class_remain)

n_db = length(imdb_in);
assert(n_db == length(roidb_in));

imdb_out = cell(1, n_db);
roidb_out = cell(1, n_db);
for i_db = 1 : n_db

    n_img = length(roidb_in{i_db}.rois);
    class_mask = zeros(1, imdb_in{i_db}.num_classes);
    class_mask(class_remain) = 1;

    rm_idx = false(1, n_img);
    rois = roidb_in{i_db}.rois;
    for i_img = 1 : n_img
        not_mem = ~ismember(rois(i_img).class, class_remain);
        if all(not_mem)
            rm_idx(i_img) = true;
            continue;
        end
        rois(i_img).gt(not_mem) = [];
        rois(i_img).overlap(not_mem, :) = [];
        rois(i_img).overlap = ...
            bsxfun(@times, rois(i_img).overlap, class_mask);
        rois(i_img).boxes(not_mem, :) = [];
        rois(i_img).class(not_mem) = [];
    end
    rois(rm_idx) = [];

    imdb = imdb_in{i_db};
    imdb.image_ids(rm_idx) = [];
    if imdb.flip && ~isempty(imdb.flip_from)
        imdb.flip_from(rm_idx) = [];
    end
    imdb.sizes(rm_idx, :) = [];
    imdb.image_at = @(i)sprintf('%s/%s.%s',imdb.image_dir,imdb.image_ids{i},imdb.extension);
%     keep_idx = ismember(imdb.class_ids, class_remain);
%     imdb.classes = imdb.classes(keep_idx);
%     imdb.class_ids = imdb.class_ids(keep_idx);

    imdb_out{i_db} = imdb;
    roidb_out{i_db}.name = roidb_in{i_db}.name;
    roidb_out{i_db}.rois = rois;

end

% for i_db = 1 : n_db
%     imdb = imdb_out{i_db};
%     for i = 1 : length(imdb.image_ids)
%         im = imread([imdb.image_dir,'/',imdb.image_ids{i},'.jpg']);
%         assert(all(size(im, 1) == imdb.sizes(i, 1)));
%         assert(all(size(im, 2) == imdb.sizes(i, 2)));
%         fprintf('%d == %d, %d == %d.\n', size(im, 1), imdb.sizes(i, 1), size(im, 2), imdb.sizes(i, 2));
%     end
% end

return



