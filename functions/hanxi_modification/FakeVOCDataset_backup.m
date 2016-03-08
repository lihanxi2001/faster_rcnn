% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Saturday 16/01/2016 14:31.
% Last Revision: Saturday 23/01/2016 01:29.

function fake_data = FakeVOCDataset()

    %% Save the images.
    source_path = '../drivingEye_carback/';
    min_dim = 600;
    min_obj_siz = 16;
    flag_mirror = true;

    % Make a folder to store the image, as VOC does.
    save_path = [source_path, '/JPEGImages_All/'];
    if ~exist(save_path, 'dir')
        mkdir(save_path);
    end

    raw_file = [save_path, 'raw_data.mat'];
    if ~exist(raw_file, 'file')  || true

        real_data = load([source_path, 'annotation.mat']);

        empty_cell = cellfun(@isempty, real_data.bbox_all);
        real_data.bbox_all(empty_cell) = [];
        real_data.bbox_cls_all(empty_cell) = [];
        real_data.img_path_all(empty_cell) = [];

        n_person = length(real_data.bbox_all);
        n_img_all = sum(cellfun(@length, real_data.bbox_all));
        image_ids = cell(1, n_img_all);
        image_sizes = zeros(n_img_all, 2);
        person_id = zeros(1, n_img_all);
        cnt = 1;
        rois = struct('gt', {}, 'overlap', {}, 'boxes', {}, 'feat', {}, 'class', {});
        for i_person = 1 : 1 : n_person

            bbox_cls_person = real_data.bbox_cls_all{i_person};
            bbox_person = real_data.bbox_all{i_person};
            img_path_person = real_data.img_path_all{i_person};
            n_img = length(bbox_person);

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
                bbox_now{1} = bbox_person{i_img};
                if flag_mirror, bbox_now{2} = flip_bbox(bbox_now{1}, size(I)); end

                for ii = 1 : length(I_cell)
                    rois(cnt).gt = true(n_bbox, 1);
                    rois(cnt).overlap = zeros(n_bbox, 20);
                    rois(cnt).overlap(:, 7) = 1;

                    rois(cnt).boxes = bbox_now{ii}';
                    rois(cnt).feat = [];
                    rois(cnt).class = 7 * ones(n_bbox, 1);

                    person_id(cnt) = i_person;
                    img_siz = size(I_cell{ii});
                    image_sizes(cnt, :) = img_siz(1 : 2);
                    image_id_now = sprintf('%06d', cnt);
                    image_ids{cnt} = image_id_now;
                    imwrite(I_cell{ii}, [save_path, '/', image_id_now, '.jpg']);
                    cnt = cnt + 1;

                    % For debugging.
    %                 show_boxes(I_cell{ii}, bbox_now{ii});
                end
                fprintf('done.\n');

            end

        end

        save(raw_file, 'image_ids', 'person_id', 'rois', 'image_sizes', '-v7.3');

    else

        load(raw_file);

    end


    %% Save as a VOC dataset.
    voc_data = [];
    voc_data = Dataset.voc0712_trainval(voc_data, 'train', false);
    voc_data = Dataset.voc2007_test(voc_data, 'test', false);

    n_person = length(unique(person_id));
    rand_idx = randperm(n_person);
    trn1_person_idx = rand_idx(1 : round(0.3 * n_person));
    trn2_person_idx = rand_idx(round(0.3 * n_person) + 1 : round(0.6 * n_person));
    tst_person_idx = rand_idx(round(0.6 * n_person) + 1 : end);

    trn_idx{1} = ismember(person_id, trn1_person_idx);
    trn_idx{2} = ismember(person_id, trn2_person_idx);
    tst_idx = ismember(person_id, tst_person_idx);

    fake_data = voc_data;

    for i_db = 1 : 2
        imdb_now = fake_data.imdb_train{i_db};
        imdb_now.image_dir = save_path;
        imdb_now.image_ids = image_ids(trn_idx{i_db});
        imdb_now.flip = 0;
        imdb_now.flip_from = zeros(length(imdb_now.image_ids), 1);
        imdb_now.sizes = image_sizes(trn_idx{i_db}, :);
        imdb_now.image_at = @(i)sprintf('%s/%s.%s',imdb_now.image_dir,imdb_now.image_ids{i},imdb_now.extension);
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
    imdb_now.image_at = @(i)sprintf('%s/%s.%s',imdb_now.image_dir,imdb_now.image_ids{i},imdb_now.extension);
    fake_data.imdb_test = imdb_now;

    roidb_now = fake_data.roidb_test;
    roidb_now.rois = rois(tst_idx);
    fake_data.roidb_test = roidb_now;

    save([save_path, 'carback_voc_data.mat'], 'fake_data', '-v7.3');

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
