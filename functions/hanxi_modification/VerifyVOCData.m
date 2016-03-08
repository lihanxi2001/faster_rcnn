% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Friday 29/01/2016 13:50.
% Last Revision: Friday 29/01/2016 13:50.

function VerifyVOCData()

    clc
    root_path = '/media/hanxi/m2_SSD/Kitti/';
    R = 0.8; save_path_sub = 'voc_format_dataset';
    save_path = [root_path, '/', save_path_sub, '/'];
    save_name = [save_path, 'kitti_voc_data_r', num2str(round(100 * R)), '.mat'];
    load(save_name);% 'fake_data'
%     obj_names = {'Car', 'Cyclist', 'Pedestrian', ...
%                 'Truck', 'Van', 'Person_sitting', 'Tram'};
    obj_color = {'r', 'g', 'b', 'y', 'k', 'c', 'm'};

    num_trn = length(fake_data.imdb_train);

    for i = 1 : num_trn
        imdb_now = fake_data.imdb_train{i};
        roi_now = fake_data.roidb_train{i};
        num_img = length(imdb_now.image_ids);
        for i_img = 1 : 20 : num_img
            img_name = sprintf('%s/%s',imdb_now.image_dir,imdb_now.image_ids{i_img});
            I = imread(img_name);
            show_boxes(I, roi_now.rois(i_img).boxes', roi_now.rois(i_img).class, obj_color);
        end
    end

    imdb_now = fake_data.imdb_test;
    roi_now = fake_data.roidb_test;
    num_img = length(imdb_now.image_ids);
    for i_img = 1 : 20 : num_img
        img_name = sprintf('%s/%s',imdb_now.image_dir,imdb_now.image_ids{i_img});
        I = imread(img_name);
        show_boxes(I, roi_now.rois(i_img).boxes', roi_now.rois(i_img).class, obj_color);
    end

end

%% *********************************************************************** %%
function show_boxes(I, bbox, bbox_cls, obj_color)

    if isempty(bbox), return; end

    figure(1);%set(gcf, 'visible', 'off');
    imagesc(I);axis equal;hold on;
    for i_bbox = 1 : size(bbox, 2)
        bbox_now = bbox(:, i_bbox);
        c = obj_color{bbox_cls(i_bbox)};
        rectangle('Position', [bbox_now(1), bbox_now(2), ...
            bbox_now(3) - bbox_now(1) + 1, bbox_now(4) - bbox_now(2) + 1], ...
                            'EdgeColor', c);
    end
    hold off;
    pause(0.1);clf;

end


