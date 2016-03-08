% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Saturday 16/01/2016 14:31.
% Last Revision: Saturday 23/01/2016 01:29.

function FakeVOCDataset(db_name, main_folder, save_folder_sub, voc_data, obj_names, R)

%     db_name = 'carback';
%     main_folder = '/media/Data/work_data/drivingEye_carback/';
%     save_folder_sub = 'JPEGImages_Batch1to12';

    if ~exist('R', 'var'), R = 0.8; end
    
    min_dim = 600;
    min_obj_siz = 16;
    flag_mirror = true;
    num_class = length(obj_names);

    %% Save the images.
    % Make a folder to store the image, as VOC does.
    save_folder = [main_folder, '/', save_folder_sub, '/'];
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end

    raw_file = [save_folder, 'raw_data.mat'];
    if ~exist(raw_file, 'file') || true

        real_data = load([save_folder, db_name, '_annotation.mat']);

        empty_cell = cellfun(@isempty, real_data.bbox_all);
        real_data.bbox_all(empty_cell) = [];
        real_data.bbox_cls_all(empty_cell) = [];
        real_data.img_path_all(empty_cell) = [];

        n_person = length(real_data.bbox_all);

        image_ids_cell = cell(1, n_person);
        image_sizes_cell = cell(1, n_person);
        person_id_cell = cell(1, n_person);
        rois_cell = cell(1, n_person);

        n_img_person = zeros(1, n_person);
        parfor i_person = 1 : n_person

            bbox_cls_person = real_data.bbox_cls_all{i_person};
            bbox_person = real_data.bbox_all{i_person};
            img_path_person = real_data.img_path_all{i_person};
            n_img = length(bbox_person);
            r = (1 + flag_mirror);
            image_ids = cell(1, n_img * r);
            image_sizes = zeros(n_img * r, 2);
            person_id = zeros(1, n_img * r);
            rois = struct('gt', cell(1, n_img * r), 'overlap', cell(1, n_img * r), ...
                'boxes', cell(1, n_img * r), 'feat', cell(1, n_img * r), 'class', cell(1, n_img * r));
            cnt = 1;
            for i_img = 1 : n_img

                fprintf('Handling person %d, image %d, ... ', i_person, i_img);
                I = imread(img_path_person{i_img});
                [I, bbox_person{i_img}] = resize_all(I, bbox_person{i_img}, min_dim);

                bbox_siz = [bbox_person{i_img}(3, :) - bbox_person{i_img}(1, :) + 1; ...
                            bbox_person{i_img}(4, :) - bbox_person{i_img}(2, :) + 1];
                bbox_ar = (bbox_person{i_img}(3, :) - bbox_person{i_img}(1, :) + 1) ...
                            ./ (bbox_person{i_img}(4, :) - bbox_person{i_img}(2, :) + 1);

                valid_idx = bbox_cls_person{i_img} == 7 & min(bbox_siz, [], 1) > min_obj_siz ...
                                                & bbox_ar < 2.5 & bbox_ar > 0.4;
                if sum(valid_idx) == 0
                    fprintf('Skipped.\n');
                    continue;
                end

                g = bbox_cls_person{i_img} == 7 & ...
                    (min(bbox_siz, [], 1) <= min_obj_siz | bbox_ar > 2.5 | bbox_ar < 0.4);
                if sum(g) > 0
                    I = fill_zero(bbox_person{i_img}(:, g), I);
                end
                I_cell = []; I_cell{1} = I;
                if flag_mirror, I_cell{2} = flip(I, 2); end

                bbox_person{i_img}(:, ~valid_idx) = [];
                n_bbox = size(bbox_person{i_img}, 2);
                bbox_now = cell(1, 0);
                bbox_now{1} = bbox_person{i_img};
                if flag_mirror, bbox_now{2} = flip_bbox(bbox_now{1}, size(I)); end

                for ii = 1 : length(I_cell)
                    rois(cnt).gt = true(n_bbox, 1);
                    rois(cnt).overlap = zeros(n_bbox, num_class);
                    for i_bbox = 1 : n_bbox
                        rois(cnt).overlap(i_bbox) = 1;
                    end

                    rois(cnt).boxes = bbox_now{ii}';
                    rois(cnt).feat = [];
                    rois(cnt).class = ones(n_bbox, 1);

                    person_id(cnt) = i_person;
                    img_siz = size(I_cell{ii});
                    image_sizes(cnt, :) = img_siz(1 : 2);
                    image_id_now = sprintf('%03d_%06d', i_person, cnt);
                    image_ids{cnt} = image_id_now;
                    imwrite(I_cell{ii}, [save_folder, '/', image_id_now, '.jpg']);
                    cnt = cnt + 1;
                    % For debugging.
%                     show_boxes(I_cell{ii}, bbox_now{ii});
                end

                fprintf('done.\n');

            end
            n_img_person(i_person) = cnt - 1;

            image_ids_cell{i_person} = image_ids(1 : n_img_person(i_person));
            image_sizes_cell{i_person} = image_sizes(1 : n_img_person(i_person), :);
            person_id_cell{i_person} = person_id(1 : n_img_person(i_person));
            rois_cell{i_person} = rois(1 : n_img_person(i_person));

        end

        % Combine the results
        n_img_all = sum(n_img_person);
        image_ids_all = cell(1, n_img_all);
        image_sizes_all = zeros(n_img_all, 2);
        person_id_all = zeros(1, n_img_all);
        rois_all = struct('gt', cell(1, n_img_all), 'overlap', cell(1, n_img_all), ...
            'boxes', cell(1, n_img_all), 'feat', cell(1, n_img_all), 'class', cell(1, n_img_all));
        cumsum_n_img = [0, cumsum(n_img_person)];
        for i_person = 1 : n_person
            idx_now = cumsum_n_img(i_person) + 1 : cumsum_n_img(i_person + 1);
            image_ids_all(idx_now) = image_ids_cell{i_person};
            image_sizes_all(idx_now, :) = image_sizes_cell{i_person};
            person_id_all(idx_now) = person_id_cell{i_person};
            rois_all(idx_now) = rois_cell{i_person};
        end

        save(raw_file, 'image_ids_all', 'person_id_all', 'rois_all', 'image_sizes_all', '-v7.3');

    else

        load(raw_file);

    end


    %% Save as a VOC dataset.
    n_person = length(unique(person_id_all));
    rand_idx = randperm(n_person);
    trn1_person_idx = rand_idx(1 : round(0.5 * R * n_person));
    trn2_person_idx = rand_idx(round(0.5 * R * n_person) + 1 : round(R * n_person));
    tst_person_idx = rand_idx(round(R * n_person) + 1 : end);

    trn_idx{1} = ismember(person_id_all, trn1_person_idx);
    trn_idx{2} = ismember(person_id_all, trn2_person_idx);
    tst_idx = ismember(person_id_all, tst_person_idx);

    fake_data = voc_data;

    for i_db = 1 : 2
        imdb_now = fake_data.imdb_train{i_db};
        imdb_now.image_dir = save_folder;
        imdb_now.image_ids = image_ids_all(trn_idx{i_db});
        imdb_now.flip = 0;
        imdb_now.flip_from = zeros(length(imdb_now.image_ids), 1);
        imdb_now.sizes = image_sizes_all(trn_idx{i_db}, :);
        imdb_now.image_at = @(i)sprintf('%s/%s.%s',imdb_now.image_dir,imdb_now.image_ids{i},imdb_now.extension);
        imdb_now.classes = obj_names;
        imdb_now = ConverImdbClass(imdb_now, obj_names);
        fake_data.imdb_train{i_db} = imdb_now;

        roidb_now = fake_data.roidb_train{i_db};
        roidb_now.rois = rois_all(trn_idx{i_db});
        fake_data.roidb_train{i_db} = roidb_now;
    end

    imdb_now = fake_data.imdb_test;
    imdb_now.image_dir = save_folder;
    imdb_now.image_ids = image_ids_all(tst_idx);
    imdb_now.flip = 0;
    imdb_now.flip_from = zeros(length(imdb_now.image_ids), 1);
    imdb_now.sizes = image_sizes_all(tst_idx, :);
    imdb_now.image_at = @(i)sprintf('%s/%s.%s',imdb_now.image_dir,imdb_now.image_ids{i},imdb_now.extension);
    imdb_now.classes = obj_names;
    imdb_now = ConverImdbClass(imdb_now, obj_names);
    fake_data.imdb_test = imdb_now;

    roidb_now = fake_data.roidb_test;
    roidb_now.rois = rois_all(tst_idx);
    fake_data.roidb_test = roidb_now;

    save([save_folder, db_name, '_voc_data.mat'], 'fake_data', '-v7.3');

end

%% *********************************************************** %%
function [I, bbox] = resize_all(I, bbox, min_dim)
    img_siz = size(I);
    r = min_dim / min(img_siz(1 : 2));
    if r < 1
        I = imresize(I, r);
        bbox = round(bbox .* r);
        bbox(1 : 2, :) = max(1, bbox(1 : 2, :));
        bbox(3, :) = min(size(I, 2), bbox(3, :));
        bbox(4, :) = min(size(I, 1), bbox(4, :));
    end
end

%% ************************************************************ %%
function show_boxes(I, bbox)
    figure(1);
    imagesc(I);axis equal;hold on;
    for i_bbox = 1 : size(bbox, 2)
        bbox_now = bbox(:, i_bbox);
        rectangle('Position', [bbox_now(1), bbox_now(2), ...
            bbox_now(3) - bbox_now(1) + 1, bbox_now(4) - bbox_now(2) + 1], 'EdgeColor', 'g');
    end
    hold off;
    pause(0.1);clf;
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

%% ************************************************************************ %%
function img_fill = fill_zero(fill_box, img_ori)

    img_fill = img_ori;
    for i_fill = 1 : size(fill_box, 2)
        img_fill(fill_box(2, i_fill) : fill_box(4, i_fill), ...
            fill_box(1, i_fill) : fill_box(3, i_fill), :) = 0;
    end

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


