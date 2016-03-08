% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Friday 29/01/2016 17:18.
% Last Revision: Monday 01/02/2016 02:34.

clear; % class;
close all;
clc;
dbstop if error

%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = 1; % auto_select_gpu;
opts.per_nms_topN           = 6000;
opts.nms_overlap_thres      = 0.7;
opts.sameclass_nums_thres   = 0.3;
opts.after_nms_topN         = 300;
opts.use_gpu                = true;
opts.test_scales            = 600;
opts.cache_name             = 'final_test';
opts.thresh                 = 0.3333;

% set gpu/cpu
if exist('./flag_caffe_initialized', 'file')
    caffe.reset_all();
else
%     clear mex;
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
% model_dir = fullfile(pwd, 'output', 'faster_rcnn_final', ...
%                             'faster_rcnn_carback_ZFHalf'); %% ZF Half
model_dir = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC0712LessClass_ZF'); %% ZF
proposal_detection_model = LoadProposalDetectionModel(model_dir);

proposal_detection_model.conf_proposal.test_scales = opts.test_scales;
proposal_detection_model.conf_detection.test_scales = opts.test_scales;
if opts.use_gpu
    proposal_detection_model.conf_proposal.image_means = ...
                gpuArray(proposal_detection_model.conf_proposal.image_means);
    proposal_detection_model.conf_detection.image_means = ...
                gpuArray(proposal_detection_model.conf_detection.image_means);
end

root_folder = '/media/hanxi/m2_SSD/Kitti/';
R = 0.8;
db_folder = [root_folder, '/voc_format_dataset/'];
db_name = 'kitti';
load([db_folder, db_name, '_voc_data_r', num2str(round(100 * R)), '.mat']);% 'fake_data'

imdb_tst = fake_data.imdb_test;
num_img = 20; % length(imdb_tst.image_ids);
classes = proposal_detection_model.classes;
num_classes = length(classes);

% proposal net
rpn_net = caffe.Net(proposal_detection_model.proposal_net_def, 'test');
rpn_net.copy_from(proposal_detection_model.proposal_net);
% fast rcnn net
fast_rcnn_net = caffe.Net(proposal_detection_model.detection_net_def, 'test');
fast_rcnn_net.copy_from(proposal_detection_model.detection_net);

%% -------------------- TESTING --------------------
running_time = zeros(1, num_img);
aboxes_all = cell(num_classes, 1);
gt_all = cell(num_classes, 1);
for i = 1 : num_classes
    aboxes_all{i} = cell(num_img, 1);
    gt_all{i} = cell(num_img, 1);
end
img_name_all = cell(1, num_img);

for i_img = 1 : num_img

    img_name = sprintf('%s/%s',imdb_tst.image_dir,imdb_tst.image_ids{i_img});
    img_name_all{i_img} = img_name;
    im = imread(img_name);
    roi = fake_data.roidb_test.rois(i_img);

    if opts.use_gpu, im = gpuArray(im); end

    % test proposal
    th = tic();
    [boxes, scores] = ...
        proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im);
    t_proposal = toc(th);
    th = tic();
    aboxes = BoxesFilter([boxes, scores], ...
        opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
    t_nms = toc(th);

    % test detection
    th = tic();
    [boxes, scores] = fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, ...
        fast_rcnn_net, im, rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
                                aboxes(:, 1:4), opts.after_nms_topN);
    t_detection = toc(th);

    fprintf('%s (%dx%d): time %.3fs (resize+conv+proposal: %.3fs, nms+regionwise: %.3fs)\n', ...
        img_name, size(im, 2), size(im, 1), t_proposal + t_nms + t_detection, ...
                                t_proposal, t_nms+t_detection);
    running_time(i_img) = t_proposal + t_nms + t_detection;

    % visualize
    boxes_cell = cell(num_classes, 1);
    line_style = cell(num_classes, 1);
    for i = 1 : num_classes

        boxes_class = [boxes(:, (1 + (i - 1) * 4) : (i * 4)), scores(:, i)];
        boxes_gt = roi.boxes(roi.class == i, :);
        gt_all{i}{i_img} = boxes_gt;

        nms_idx = nms(boxes_class, opts.sameclass_nums_thres);
        high_idx = find(boxes_class(:, 5) >= opts.thresh);
        keep_idx = intersect(high_idx, nms_idx);
        aboxes_all{i}{i_img} = boxes_class(keep_idx, :);
        boxes_class = boxes_class(keep_idx, 1 : 4);

        boxes_cell{i} = [boxes_gt; boxes_class];
        line_style_now = [cell(1, size(boxes_gt, 1)), cell(1, size(boxes_class, 1))];
        line_style_now(1 : size(boxes_gt, 1)) = {'-.'};
        line_style_now(size(boxes_gt, 1) + 1 : end) = {'-'};
        line_style{i} = line_style_now;

    end
    figure(1);
    HanxiShowboxes(im, boxes_cell, classes, line_style);hold off;
    pause(0.05);clf;
    clear('im');

end


res_folder = './Exp_Result/';
if ~exist(res_folder, 'dir'), mkdir(res_folder); end
save_name = [res_folder, db_name, '_DetectRes_', 'ImgNum', num2str(num_img), ...
    '_Thresh', num2str(round(opts.thresh * 100)), '_SameClassOverlap', ...
                    num2str(round(opts.sameclass_nums_thres * 10)), '.mat'];
save(save_name, 'classes', 'aboxes_all', 'gt_all', 'img_name_all', '-v7.3');

class_pick = {'Car'; 'Truck'; 'Van'};
[ap_all, prec_all, rec_all] = ...
    DetectionResEval(db_name, num_img, classes, opts.thresh, class_pick, ...
                                    opts.sameclass_nums_thres, res_folder);

caffe.reset_all();

