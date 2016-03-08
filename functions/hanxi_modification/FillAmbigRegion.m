% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Saturday 16/01/2016 15:02.
% Last Revision: Sunday 31/01/2016 21:28.

%% ******************************************************************* %%
function FillAmbigRegion(db_name, main_folder, save_folder_sub)

%     db_name = 'carback';
%     main_folder = '/home/hanxi/work/drivingEye_carback/';
%     save_folder_sub = '/JPEGImages_Batch1to4/';

    bbox_all = cell(1, 1000);
    bbox_cls_all = cell(1, 1000);
    img_path_all = cell(1, 1000);
    main_dir = dir(main_folder);
    sub_cnt = 1;
    for i_batch = 1 : length(main_dir)

        if ~main_dir(i_batch).isdir || ...
            strcmpi(main_dir(i_batch).name, '.') || strcmpi(main_dir(i_batch).name, '..')
            continue;
        end
        batch_folder = main_dir(i_batch).name;
        batch_dir = dir([main_folder, '/', batch_folder]);

        for i_sub = 1 : length(batch_dir)

            if ~batch_dir(i_sub).isdir || ...
                strcmpi(batch_dir(i_sub).name, '.') || strcmpi(batch_dir(i_sub).name, '..')
                continue;
            end

            sub_folder = [main_folder, '/', batch_folder, '/', batch_dir(i_sub).name];
            txt_dir = dir([sub_folder, '/*.txt']);

            if isempty(txt_dir), continue; end

%             profile on
            fprintf('Now handling %s ... ', batch_dir(i_sub).name);
            txt_file = txt_dir(1).name;
            fid = fopen([sub_folder, '/', txt_file], 'r');
            tline = fgetl(fid);
            bbox_sub = zeros(4, 10000);
            bbox_cls_sub = zeros(1, 10000);
            img_path_sub = cell(1, 10000);
            cnt = 0;
            while ischar(tline) % && cnt <= 2

                img_path = regexp(tline, ...
                    [SpecialCharacterProcess(batch_dir(i_sub).name), '\\(.*\.jpg)'], 'tokens');
                box_str = regexp(tline, ...
                    '[0-9]+\s+[0-9]+\s+[0-9]+\s+[0-9]+\s+[0-9]+\s+[0-9]', 'match');
                if isempty(img_path) || isempty(box_str)
                    tline = fgetl(fid);
                    continue;
                end

                img_path = img_path{1}{1};
                img_path_now = regexprep(img_path, '\\', '/');
                if ~exist([sub_folder, '/', img_path_now], 'file')
                    tline = fgetl(fid);
                    continue;
                end

                img_path_sub{cnt + 1} = img_path_now;
                box_str = box_str{1};
                box_param = sscanf(box_str, '%d');
                bbox = box_param(1 : 4) + 1;
                bbox_sub(:, cnt + 1) = bbox([1, 3, 2, 4]);
                bbox_cls = box_param(5);
                bbox_cls(bbox_cls <= 3) = 7;
                bbox_cls(bbox_cls == 4) = -2;
                bbox_cls_sub(cnt + 1) = bbox_cls;

                tline = fgetl(fid);
                cnt = cnt + 1;

            end
            bbox_sub(:, cnt : end) = [];
            bbox_cls_sub(cnt : end) = [];
            img_path_sub(cnt : end) = [];
            fclose(fid);

            compare_res = compare_img_path(img_path_sub);
            [bbox_sub, bbox_cls_sub, img_path_sub] = ...
                        merge_info(bbox_sub, bbox_cls_sub, img_path_sub, compare_res);

            parfor j = 1 : length(bbox_sub)
%             for j = 1 : length(bbox_sub)
                img_path_sub{j} = [sub_folder, '/', img_path_sub{j}];
%                 assert(exist(img_path_sub{j}, 'file') == 2);
                I = imread(img_path_sub{j});
                keep_idx = bbox_cls_sub{j} == 7;
                fill_idx = bbox_cls_sub{j} == -2;
                I_fill = fill_random(bbox_sub{j}(:, fill_idx), bbox_sub{j}(:, keep_idx), I);
                imwrite(I_fill, img_path_sub{j});
%                 show_boxes(I, I_fill, bbox_sub{j}, bbox_cls_sub{j});
            end

            bbox_all{sub_cnt} = bbox_sub;
            bbox_cls_all{sub_cnt} = bbox_cls_sub;
            img_path_all{sub_cnt} = img_path_sub;
            sub_cnt = sub_cnt + 1;
            fprintf('done.\n');
%             profile viewer;
%             profile off;

        end

    end

    if ~exist([main_folder, save_folder_sub], 'dir')
        mkdir([main_folder, save_folder_sub]);
    end
    save([main_folder, save_folder_sub, db_name, '_annotation.mat'], ...
                'bbox_all', 'bbox_cls_all', 'img_path_all', '-v7.3');

end

%% ************************************************************************ %%
function compare_res = compare_img_path(img_path)

    compare_res = zeros(length(img_path));
    for i_path = 1 : length(img_path)
        compare_res(i_path, :) = ...
            cellfun(@(x) strcmp(x, img_path{i_path}), img_path);
    end
    compare_res = logical(unique(compare_res, 'rows'));

end

%% ************************************************************************ %%
function [bbox_merge, bbox_cls_merge, img_path_merge] = ...
                        merge_info(bbox, bbox_cls, img_path, compare_res)
    n_merge = size(compare_res, 1);
    bbox_merge = cell(1, n_merge);
    bbox_cls_merge = cell(1, n_merge);
    img_path_merge = cell(1, n_merge);
    for i = 1 : n_merge
        bbox_merge{i} = bbox(:, compare_res(i, :));
        bbox_cls_merge{i} = bbox_cls(compare_res(i, :));
        img_path_merge{i} = img_path{find(compare_res(i, :), 1, 'first')};
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
function show_boxes(I, I_fill, bbox, bbox_cls)

    figure(1);%set(gcf, 'visible', 'off');
    subplot(1, 2, 1);
    imagesc(I);axis equal;hold on;
    for i_bbox = 1 : size(bbox, 2)
        bbox_now = bbox(:, i_bbox);
        if bbox_cls(i_bbox) > 0
            c = 'g';
        else
            c = 'r';
        end
        rectangle('Position', [bbox_now(1), bbox_now(2), ...
            bbox_now(3) - bbox_now(1) + 1, bbox_now(4) - bbox_now(2) + 1], 'EdgeColor', c);
    end
    hold off;
    figure(1);subplot(1, 2, 2);
    imagesc(I_fill);axis equal;
    pause(0.05);clf;

end
