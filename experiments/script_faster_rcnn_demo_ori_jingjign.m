function script_faster_rcnn_demo_ori()
close all;
clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
ROI = false;

opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);
opts.per_nms_topN           = 3000;
opts.nms_overlap_thres      = 0.8;
opts.sameclass_nums_thres   = 0.3;
opts.after_nms_topN         = 300;
opts.use_gpu                = true;
if ROI == false
    opts.test_scales            = 600;
else
    opts.test_scales            = 400;
end

%% -------------------- INIT_MODEL --------------------
model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC0712_vgg_16layers'); %% VGG-16
% model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC0712_ZF'); %% ZF
%  model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', '5w');
% model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', '10w');
% model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', '10w-finetune');
% model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', 'conv2+48');
% model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', 'conv2+48-1');
% model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', 'kitti-review');
% model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', 'conv3_4096_4096_base');
% model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', 'conv3+512+512');
% model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', 'CONV2+512+512');
% model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', 'kitti');
% model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', 'REVIEW');

proposal_detection_model    = load_proposal_detection_model(model_dir);

proposal_detection_model.conf_proposal.test_scales = opts.test_scales;
proposal_detection_model.conf_detection.test_scales = opts.test_scales;
if opts.use_gpu
    proposal_detection_model.conf_proposal.image_means = gpuArray(proposal_detection_model.conf_proposal.image_means);
    proposal_detection_model.conf_detection.image_means = gpuArray(proposal_detection_model.conf_detection.image_means);
end

% caffe.init_log(fullfile(pwd, 'caffe_log'));
% proposal net
rpn_net = caffe.Net(proposal_detection_model.proposal_net_def, 'test');
rpn_net.copy_from(proposal_detection_model.proposal_net);
% fast rcnn net
fast_rcnn_net = caffe.Net(proposal_detection_model.detection_net_def, 'test');
fast_rcnn_net.copy_from(proposal_detection_model.detection_net);

% set gpu/cpu
if opts.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end

%% -------------------- WARM UP --------------------
% the first run will be slower; use an empty image to warm up

for j = 1:2 % we warm up 2 times
    im = uint8(ones(375, 500, 3)*128);
    if opts.use_gpu
        im = gpuArray(im);
    end
    [boxes, scores]             = proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im);
    aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
    if proposal_detection_model.is_share_feature
        [boxes, scores]             = fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
            aboxes(:, 1:4), opts.after_nms_topN);
    else
        [boxes, scores]             = fast_rcnn_im_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            aboxes(:, 1:4), opts.after_nms_topN);
    end
end

%% -------------------- TESTING --------------------
%im_names = {'1.jpg'};
%im_names = {'1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg','10.jpg','11.jpg','12.jpg','13.jpg','14.jpg','15.jpg','16.jpg','17.jpg','18.jpg','19.jpg','20.jpg'};

video_file = './TestImages/jingjing.avi';
video = VideoReader(video_file);
frame_number = floor(video.Duration * video.FrameRate);

count = 1;
running_time = [];
for j=1:frame_number%length(im_names)

    im=read(video,j);


  if mod(j,30)==0
    im = imresize(im,[360,640]);
 %   im = imread(fullfile(pwd, im_names{j}));

    if ROI == true
       im = im(121:360,:,:);
    end

    if opts.use_gpu
        im = gpuArray(im);
    end

    % test proposal
 %   th = tic();
    [boxes, scores]             = proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im);
%    t_proposal = toc(th);
%    th = tic();
    aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
%    t_nms = toc(th);

    %plot proposal
   %showPospal(im,aboxes);

    % test detection
 %   th = tic();
    if proposal_detection_model.is_share_feature
        [boxes, scores]             = fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
            aboxes(:, 1:4), opts.after_nms_topN);
    else
        [boxes, scores]             = fast_rcnn_im_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            aboxes(:, 1:4), opts.after_nms_topN);
    end
%    t_detection = toc(th);

%     fprintf('%s (%dx%d): time %.3fs (resize+conv+proposal: %.3fs, nms: %.3fs,detection: %.3fs)\n', file, ...
%         size(im, 2), size(im, 1), t_proposal + t_nms + t_detection, t_proposal, t_nms,t_detection);
%     running_time(end+1) = t_proposal + t_nms + t_detection;

    % visualize
    classes = proposal_detection_model.classes;
    boxes_cell = cell(length(classes), 1);
     thres = 0.8;
    for i = 1:length(boxes_cell)
        boxes_cell{i} = [boxes(:, (1+(i-1)*4):(i*4)), scores(:, i)];
        boxes_cell{i} = boxes_cell{i}(nms(boxes_cell{i}, opts.sameclass_nums_thres), :);
        I = boxes_cell{i}(:, 5) >= thres;
        boxes_cell{i} = boxes_cell{i}(I, :);
    end

%    figure;
%    showboxes(im, boxes_cell, classes, 'voc');


    figure(1);set(gcf, 'visible', 'off');
    showboxes(im, boxes_cell, classes, 'voc');axis equal;
    path = strcat('E:\result\', num2str(count),'.jpg');
    saveas(gcf, path);
    count = count + 1;
     %pause(1);
     clf;
 end
end
% fprintf('mean time: %.3fs\n', mean(running_time));

caffe.reset_all();
clear mex;

end
function showPospal(im,aboxes)
    figure;
    imshow(im);
    x = aboxes(:,1);
    y = aboxes(:,2);
    w = aboxes(:,3)-aboxes(:,1);
    h = aboxes(:,4)-aboxes(:,2);
    for i = 1:15
        hold on;
         rectangle('Position', [x(i) y(i) w(i) h(i)],'edgecolor','r');
    end
end

function proposal_detection_model = load_proposal_detection_model(model_dir)
    ld                          = load(fullfile(model_dir, 'model'));
    proposal_detection_model    = ld.proposal_detection_model;
    clear ld;

    proposal_detection_model.proposal_net_def ...
                                = fullfile(model_dir, proposal_detection_model.proposal_net_def);
    proposal_detection_model.proposal_net ...
                                = fullfile(model_dir, proposal_detection_model.proposal_net);
    proposal_detection_model.detection_net_def ...
                                = fullfile(model_dir, proposal_detection_model.detection_net_def);
    proposal_detection_model.detection_net ...
                                = fullfile(model_dir, proposal_detection_model.detection_net);

end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    if per_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        aboxes = aboxes(nms(aboxes, nms_overlap_thres, use_gpu), :);
    end
    if after_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
    end
end
