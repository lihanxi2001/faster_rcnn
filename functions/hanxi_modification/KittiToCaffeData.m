% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Saturday 23/01/2016 01:26.
% Last Revision: Thursday 28/01/2016 10:15.

function KittiToCaffeData()

    clear; close all; clc;
    disp('======= KITTI Data to Caffe Format =======');

    % options
    root_path = '/media/hanxi/m2_SSD/Kitti/';
    R = 0.8;
    save_path_sub = 'voc_format_dataset';

    obj_names = {'Car', 'Cyclist', 'Pedestrian', ...
                'Truck', 'Van', 'Person_sitting', 'Tram'};
    num_class = length(obj_names);
    obj_color = {'r', 'g', 'b', 'y', 'k', 'c', 'm'};
    data_set = 'training';
    min_obj_siz = 16;
    flag_flip = true;
    flip_name = {'_ori', '_flip'};

    % get sub-directories
    cam = 2; % 2 = left color camera
    img_dir = fullfile(root_path, [data_set, '/image_', num2str(cam)]);
    label_dir = fullfile(root_path, [data_set, '/label_', num2str(cam)]);

    % Make a folder to store the image, as VOC does.
    save_path = [root_path, '/', save_path_sub, '/'];
    if ~exist(save_path, 'dir'), mkdir(save_path); end

    % main loop
    raw_file = [save_path, 'raw_data.mat'];
    if ~exist(raw_file, 'file') || true

        n_img = length(dir(fullfile(img_dir, '*.png')));
        image_ids = cell(1, n_img);
        image_sizes = cell(1, n_img);
        rois = struct('gt', cell(1, n_img), 'overlap', cell(1, n_img), ...
            'boxes', cell(1, n_img), 'feat', cell(1, n_img), 'class', cell(1, n_img));
        if flag_flip
            image_ids_flip = cell(1, n_img);
            image_sizes_flip = cell(1, n_img);
            rois_flip = struct('gt', cell(1, n_img), 'overlap', cell(1, n_img), ...
                'boxes', cell(1, n_img), 'feat', cell(1, n_img), 'class', cell(1, n_img));
        end
        parfor img_idx = 1 : n_img

            fprintf('Handling image %d, ... ', img_idx - 1);

            [bbox, bbox_cls, ~, img_path] = ...
                    readLabels(label_dir, img_idx - 1, img_dir, obj_names);

            I = imread(img_path);
            bbox = normalize_bbox(bbox, size(I));

            bbox_siz = [bbox(3, :) - bbox(1, :) + 1; bbox(4, :) - bbox(2, :) + 1];
            small_idx = min(bbox_siz, [], 1) < min_obj_siz;
            bbox_cls(small_idx) = -2;

            keep_idx = bbox_cls >= 1 & bbox_cls <= length(obj_names);
            fill_idx = bbox_cls == -2;
            I_fill = fill_random(bbox(:, fill_idx), bbox(:, keep_idx), I);

            rm_idx = ~keep_idx;
            bbox(:, rm_idx) = [];
            bbox_cls(rm_idx) = [];
            n_bbox = size(bbox, 2);

            rois(img_idx).gt = true(n_bbox, 1);
            rois(img_idx).overlap = zeros(n_bbox, num_class);
            for i_bbox = 1 : n_bbox
                rois(img_idx).overlap(i_bbox, bbox_cls(i_bbox)) = 1;
            end
            rois(img_idx).boxes = bbox';
            rois(img_idx).feat = [];
            rois(img_idx).class = bbox_cls(:);

            img_siz = size(I_fill);
            image_sizes{img_idx} = img_siz(1 : 2);

            image_id_now = sprintf('%06d%s.jpg', img_idx, flip_name{1});
            imwrite(I_fill, [save_path, image_id_now]);
            image_ids{img_idx} = image_id_now;
            % For debugging.
%             show_boxes(I_fill, bbox, bbox_cls, obj_color);

            if flag_flip
                I_flip = flip(I_fill, 2);
                bbox = flip_bbox(bbox, size(I));
                rois_flip(img_idx).gt = true(n_bbox, 1);
                rois_flip(img_idx).overlap = zeros(n_bbox, num_class);
                for i_bbox = 1 : n_bbox
                    rois_flip(img_idx).overlap(i_bbox, bbox_cls(i_bbox)) = 1;
                end
                rois_flip(img_idx).boxes = bbox';
                rois_flip(img_idx).feat = [];
                rois_flip(img_idx).class = bbox_cls(:);

                img_siz = size(I_flip);
                image_sizes_flip{img_idx} = img_siz(1 : 2);

                image_id_now = sprintf('%06d%s.jpg', img_idx, flip_name{2});
                imwrite(I_flip, [save_path, image_id_now]);
                image_ids_flip{img_idx} = image_id_now;
                % For debugging.
%                 show_boxes(I_flip, bbox, bbox_cls, obj_color);
            end
%
            fprintf('done.\n');
%
        end
%
        if flag_flip
            image_ids = [image_ids, image_ids_flip];
            image_sizes = [image_sizes, image_sizes_flip];
            rois = [rois, rois_flip];
        end

        image_ids = image_ids(:);
        image_sizes = cell2mat(image_sizes(:));
        rois = rois(:);
        save(raw_file, 'image_ids', 'rois', 'image_sizes', '-v7.3');

    else

        load(raw_file);

    end

    %% Save as a VOC dataset.
    load([root_path, 'voc_data.mat']);

    n_img = length(image_ids);
    rand_idx = randperm(n_img);
    trn_idx{1} = rand_idx(1 : round(R / 2 * n_img));
    trn_idx{2}= rand_idx(round(R / 2 * n_img) + 1 : round(R * n_img));
    tst_idx = rand_idx(round(R * n_img) + 1 : end);

    fake_data = voc_data;
    for i_db = 1 : 2
        imdb_now = fake_data.imdb_train{i_db};
        imdb_now.image_dir = save_path;
        imdb_now.image_ids = image_ids(trn_idx{i_db});
        imdb_now.flip = 0;
        imdb_now.flip_from = zeros(length(imdb_now.image_ids), 1);
        imdb_now.sizes = image_sizes(trn_idx{i_db}, :);
        imdb_now.image_at = @(i)sprintf('%s/%s',imdb_now.image_dir,imdb_now.image_ids{i});
        imdb_now.classes = obj_names;
        imdb_now = ConverImdbClass(imdb_now, obj_names);
        fake_data.imdb_train{i_db} = imdb_now;

        roidb_now = fake_data.roidb_train{i_db};
        roidb_now.rois = rois(trn_idx{i_db});
        fake_data.roidb_train{i_db} = roidb_now;
    end

    imdb_now = fake_data.imdb_test;
    imdb_now.image_dir = save_path;
    imdb_now.image_ids = image_ids(tst_idx);
    imdb_now.flip = 0;
    imdb_now.flip_from = zeros(length(imdb_now.image_ids), 1);
    imdb_now.sizes = image_sizes(tst_idx, :);
    imdb_now.image_at = @(i)sprintf('%s/%s',imdb_now.image_dir,imdb_now.image_ids{i});
    imdb_now = ConverImdbClass(imdb_now, obj_names);
    fake_data.imdb_test = imdb_now;

    roidb_now = fake_data.roidb_test;
    roidb_now.rois = rois(tst_idx);
    fake_data.roidb_test = roidb_now;

    save([save_path, 'kitti_voc_data_r', num2str(round(100 * R)), '.mat'], 'fake_data', '-v7.3');

end


%% ********************************************************************* %%
function [bbox, bbox_cls, obj_direct, img_path] = ...
                        readLabels(label_dir, img_idx, img_dir, obj_names)

    % parse input file
    fid = fopen(sprintf('%s/%06d.txt',label_dir,img_idx),'r');
    C = textscan(fid,'%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f',...
                                                'delimiter', ' ');
    fclose(fid);
    img_path = sprintf('%s/%06d.png', img_dir, img_idx);

    % for all objects do
    num_obj = numel(C{1});
    bbox = zeros(4, num_obj);
    bbox_cls = zeros(1, num_obj);
    obj_direct = zeros(1, num_obj);
    for o = 1 : num_obj

        % extract label considering the truncation, occlusion
        type_name = C{1}(o); % for converting: cell -> string
        type_name = type_name{1};
        trunc_r = C{2}(o);
        occ_state = C{3}(o);
        name_match = cellfun(@(x) strcmpi(x, type_name), obj_names);
        if sum(name_match) > 0
          bbox_cls(o) = find(name_match, 1, 'first');
        end
        if trunc_r > 0.2, bbox_cls(o) = -2; end
        if occ_state >= 2, bbox_cls(o) = -2; end

        % extract 2D bounding box in 0-based coordinates
        bbox(:, o) = 1 + [C{5}(o); C{6}(o); C{7}(o); C{8}(o)];

        % the orientation of the car
        obj_direct(o) = C{4}(o);

    end

end

%% ************************************************************************ %%
function img_fill = fill_random(fill_box, keep_box, img_ori)

    img_siz = size(img_ori);
    rand_map = zeros(size(img_ori));
    N = numel(img_ori);
    N = floor(N / 256) * 256;
    img_idx = reshape(1 : N, 256, []);
    img_idx = img_idx(:, randperm(size(img_idx, 2)));
    rand_map(1 : N) = img_ori(img_idx(:));

%     dist = CalcBoxOverlap(keep_box, fill_box, 'int_uni');
    keep_map = zeros(img_siz(1 : 2));
    for i_keep = 1 : size(keep_box, 2)
        keep_map(keep_box(2, i_keep) : keep_box(4, i_keep), ...
            keep_box(1, i_keep) : keep_box(3, i_keep)) = 1;
    end
    keep_map = logical(keep_map);
    fill_map = zeros(img_siz(1 : 2));
    for i_fill = 1 : size(fill_box, 2)
        fill_map(fill_box(2, i_fill) : fill_box(4, i_fill), ...
            fill_box(1, i_fill) : fill_box(3, i_fill)) = 1;
    end
    fill_map = logical(fill_map);

    fill_map = fill_map & (~keep_map);
    if length(img_siz) == 3
        fill_map = repmat(fill_map, 1, 1, 3);
    end

    img_fill = img_ori;
    img_fill(fill_map) = rand_map(fill_map);

end

%% *********************************************************************** %%
function show_boxes(I, bbox, bbox_cls, obj_color)

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
    pause(0.05);clf;

end

%% *********************************************************************** %%
function bbox = normalize_bbox(bbox, img_siz)

    bbox(1, :) = max(round(bbox(1, :)), 1);
    bbox(2, :) = max(round(bbox(2, :)), 1);
    bbox(3, :) = min(round(bbox(3, :)), img_siz(2));
    bbox(4, :) = min(round(bbox(4, :)), img_siz(1));

end

%% ************************************************************ %%
function bbox_out = flip_bbox(bbox, img_siz)

    bbox_out = bbox;
    bbox_out([1, 3], :) = img_siz(2) - bbox([1, 3], :);
%     bbox_out([2, 4], :) = img_siz(1) - bbox([2, 4], :);
    comp_siz = [img_siz(2), img_siz(1), img_siz(2), img_siz(1)];
    for i = 1 : size(bbox_out, 1)
        bbox_out(i, :) = max(1, min(comp_siz(i), bbox_out(i, :)));
    end
    bbox_out = bbox_out([3, 2, 1, 4], :);

end

%% ************************************************************ %%
function dst = ConverImdbClass(src, obj_names)

    dst = src;
    dst.classes = obj_names;
    dst.num_classes = length(obj_names);
    dst.class_ids = 1 : length(obj_names);
    dst.class_to_id = containers.Map(obj_names, dst.class_ids);
    dst.details.VOCopts.classes = obj_names;
    dst.details.VOCopts.nclasses = length(obj_names);

end

