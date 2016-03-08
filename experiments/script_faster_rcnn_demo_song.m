% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Sunday 21/02/2016 17:12.
% Last Revision: Tuesday 23/02/2016 12:33.

clear; % class;
close all;
clc;
dbstop if error

%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = 1; % auto_select_gpu;
opts.per_nms_topN           = 3000;
opts.nms_overlap_thres      = 0.75;
opts.sameclass_nums_thres   = 0.3;
opts.show_score_threh       = 0.9;
opts.after_nms_topN         = 600;
opts.use_gpu                = true;
opts.test_scales            = 200;
opts.cache_name             = 'final_test';

% set gpu/cpu
if exist('./flag_caffe_initialized', 'file')
    caffe.reset_all();
else
    clear is_valid_handle; % to clear init_key
    active_caffe_mex(opts.gpu_id, opts.caffe_version);
    if opts.use_gpu
        caffe.set_mode_gpu();
    else
        caffe.set_mode_cpu();
    end
    fid = fopen('./flag_caffe_initialized', 'w');
    fprintf(fid, 'caffe has been initialized!\n');
    fclose(fid);
end

%% -------------------- INIT_MODEL --------------------
% model_dir = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC0712LessClass_ZF'); %% ZF
model_dir = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC0712_vgg_16layers'); %% VGG-16
% model_dir = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC0712_ZF'); %% ZF
proposal_detection_model = LoadProposalDetectionModel(model_dir);

proposal_detection_model.conf_proposal.test_scales = opts.test_scales;
proposal_detection_model.conf_detection.test_scales = opts.test_scales;
if opts.use_gpu
    proposal_detection_model.conf_proposal.image_means = ...
                gpuArray(proposal_detection_model.conf_proposal.image_means);
    proposal_detection_model.conf_detection.image_means = ...
                gpuArray(proposal_detection_model.conf_detection.image_means);
end

res_folder = './Exp_Result/';
if ~exist(res_folder, 'dir'), mkdir(res_folder); end
img_res_folder = [res_folder, 'res_img/'];
if ~exist(img_res_folder, 'dir'), mkdir(img_res_folder); end

class_now = 'person'; % imdb_tst.classes;
class_idx = find(cellfun(@(x) strcmp(x, class_now), ...
            proposal_detection_model.classes));
assert(~isempty(class_idx));
num_classes = 1;

% proposal net
rpn_net = caffe.Net(proposal_detection_model.proposal_net_def, 'test');
rpn_net.copy_from(proposal_detection_model.proposal_net);
% fast rcnn net
fast_rcnn_net = caffe.Net(proposal_detection_model.detection_net_def, 'test');
fast_rcnn_net.copy_from(proposal_detection_model.detection_net);

%% -------------------- TESTING --------------------
% cam = webcam;
% num_img = 400;

video_file = './TestImages/jingjing.avi';
video = VideoReader(video_file);
num_img = floor(video.Duration * video.FrameRate);
running_time = zeros(1, num_img);
T = 8; Z = 0.333;
M = 8;
T_smooth = 3;
flag_track = false;
flag_smooth = true;
tracker_cell = cell(1, M);
points_cell = cell(1, M);
validity_r = ones(1, M);
for i = 1 : M
    tracker_cell{i} = vision.PointTracker('MaxBidirectionalError', 1);
%     validity_r(i) = 0
end
n_boxes = M;

boxes_cell = cell(1, 0);
scores_cell = cell(1, 0);
f_cell = cell(1, 0);

for i_f = 1 : num_img

%     im = imresize(snapshot(cam), 0.333);
    im = read(video, i_f);

    if opts.use_gpu, im_gpu = gpuArray(im); end

    if ~flag_track || rem(i_f, T) == 0 || i_f == 1 || n_boxes == 0 || min(validity_r(1 : n_boxes)) < Z
        % test proposal
        th = tic();
        [boxes, scores] = ...
            proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im_gpu);
        t_proposal = toc(th);
        th = tic();
        aboxes = BoxesFilter([boxes, scores], ...
            opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
        t_nms = toc(th);

        % test detection
        th = tic();
        [boxes, scores] = fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, ...
            fast_rcnn_net, im_gpu, rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
                                    aboxes(:, 1:4), opts.after_nms_topN);
        t_detection = toc(th);
        scores = scores(:, class_idx);
        boxes = boxes(:, (1 + (class_idx - 1) * 4) : (class_idx * 4));

        if flag_smooth
            if i_f > 2
                f_diff = cellfun(@(x) i_f - x, f_cell, 'UniformOutput', false);
                [smooth_score] = SmoothBoxOverTime(boxes', scores, ...
                    cell2mat(boxes_cell), cell2mat(scores_cell), cell2mat(f_diff), 3, 0.25, T_smooth);
            end
            boxes_cell{end + 1} = boxes';
            scores_cell{end + 1} = scores(:)';
            f_cell{end + 1} = i_f * ones(1, numel(scores));
            if length(boxes_cell) > T_smooth
                boxes_cell(1) = [];
                scores_cell(1) = [];
                f_cell(1) = [];
            end
            if i_f > 2
                scores = smooth_score;
%                 boxes = smooth_boxes;
            end
        end

        fprintf('Frame-%d (%dx%d): time %.3fs (resize+conv+proposal: %.3fs, nms+regionwise: %.3fs)\n', ...
            i_f, size(im, 2), size(im, 1), t_proposal + t_nms + t_detection, ...
                                    t_proposal, t_nms+t_detection);
        running_time(i_f) = t_proposal + t_nms + t_detection;

        % Filter boxes and show.
        box_det = boxes;
        nms_idx = nms([box_det, scores], opts.sameclass_nums_thres);
        high_idx = find(scores >= opts.show_score_threh);
        rem_idx = intersect(high_idx, nms_idx);
        n_boxes = numel(rem_idx);
        if n_boxes > M
            [~, sort_idx] = sort(scores, 'descend');
            rem_idx = rem_idx(sort_idx(1 : M));
            n_boxes = M;
        end
        box_det = box_det(rem_idx, :);

        if flag_track
            if i_f == 1
                 for i_box = 1 : M
                    initialize(tracker_cell{i_box}, single(randi(10, [10, 2])), im);
                 end
            end

            for i_box = 1 : n_boxes
                roi = round(box_det(i_box, :));
                roi(3) = roi(3) - roi(1) + 1;
                roi(4) = roi(4) - roi(2) + 1;
                points = detectMinEigenFeatures(rgb2gray(im), ...
                                        'ROI', roi, 'MinQuality', 0.05);
                points_cell{i_box} = points.Location;
                setPoints(tracker_cell{i_box}, points_cell{i_box});
            end
            validity_r(:) = 1;
        end

    else
        tic
        for i_box = 1 : n_boxes

            old_points = points_cell{i_box};
            [points_cell{i_box}, validity] = step(tracker_cell{i_box}, im);
            validity_r(i_box) = sum(validity) / numel(validity);

            if sum(validity) >= 5
                xform = estimateGeometricTransform(old_points(validity, :), ...
                    points_cell{i_box}(validity, :), 'similarity', 'MaxDistance', 4);
                V = transformPointsForward(xform, ...
                    [box_det(i_box, 1 : 2); box_det(i_box, 3 : 4)]);
                V(1, 1) = max(1, V(1, 1));
                V(1, 2) = max(1, V(1, 2));
                V(2, 1) = max(V(1, 1) + 1, min(size(im, 2), V(2, 1)));
                V(2, 2) = max(V(1, 2) + 1, min(size(im, 1), V(2, 2)));
                box_det(i_box, :) = [V(1, :), V(2, :)];
            else
                validity_r(i_box) = 0;
            end
        end
        tt = toc;
        fprintf('Frame-%d (%dx%d): time %.3fs\n', ...
                    i_f, size(im, 2), size(im, 1), tt);
        running_time(i_f) = tt;
    end

    % visualize
    if i_f > 1
        line_style_now = cell(1, n_boxes);
        line_style_now(:) = {'-'};
        h = figure(1); % set(gcf, 'visible', 'off');
        set(h, 'Position', [300, 800, 1280, 720]);
        HanxiShowboxes(im, {box_det(validity_r(1 : n_boxes) > 0, :)}, class_now, ...
                        {line_style_now(validity_r(1 : n_boxes) > 0)});hold off;
        title('Driving-Eye Demo','FontSize', 30);
%         pause(0.015);clf;
    end
    clear('im');

end

avg_time = mean(running_time(2 : end));
fprintf('The average FPS is %2.3f.\n', 1 / avg_time);
close all;
clear cam;

