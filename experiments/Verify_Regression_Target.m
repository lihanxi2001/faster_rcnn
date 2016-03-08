% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Thursday 31/12/2015 12:19.
% Last Revision: Thursday 31/12/2015 12:19.

clc;
clear;
close all;
dbstop if error;
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));

roidb_name = 'voc0712_lessclass_image_roidb_train.mat';
load(roidb_name); % image_roidb_train and conf

num_img = length(image_roidb_train);
rand_idx = randperm(num_img, 10);
class_remain = [2, 6, 7, 14, 15];
class_color = {'r', 'g', 'b', 'y', 'k'};

for i = 1 : length(rand_idx)

    img_idx = rand_idx(i);
    im_roi_now = image_roidb_train(img_idx);

    im = imread(im_roi_now.image_path);
    im_size = size(im);
    reg_targets = im_roi_now.bbox_targets{1};
    [anchors, im_scales] = proposal_locate_anchors(conf, im_roi_now.im_size);

    valid_idx = reg_targets(:, 1) > 0;
    anchors = anchors{1}(valid_idx, :);
    box_delta = full(reg_targets(valid_idx, 2 : end));
    box_delta = bsxfun(@times, box_delta, bbox_stds);
    box_delta = bsxfun(@plus, box_delta, bbox_means);
    box_label = full(reg_targets(valid_idx, 1));

    scaled_im_size = round(im_size(1 : 2) * im_scales{1});

    pred_boxes = fast_rcnn_bbox_transform_inv(anchors, box_delta);
    pred_boxes = bsxfun(@times, pred_boxes - 1, ([im_size(2), im_size(1), ...
        im_size(2), im_size(1)] - 1) ./ ([scaled_im_size(2), scaled_im_size(1), ...
                                scaled_im_size(2), scaled_im_size(1)] - 1)) + 1;
    pred_boxes = ClipBoxes(pred_boxes, size(im, 2), size(im, 1));
    pred_boxes = unique(pred_boxes, 'rows');
    n_box = size(pred_boxes, 1);

    figure(1);imagesc(im);axis equal;
    for j = 1 : n_box
        rectsLTWH = RectLTRB2LTWH(pred_boxes(j, :));
        i_class = find(class_remain == box_label(j), 1, 'first');
        rectangle('Position', rectsLTWH, 'LineWidth', 1, 'EdgeColor', class_color{i_class});
    end
    pause(1);clf;

end


