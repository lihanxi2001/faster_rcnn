function script_faster_rcnn_carback_ZF()

% script_faster_rcnn_VOC0712_ZF()
% Faster rcnn training and testing with Zeiler & Fergus model
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

clc;
clear mex;
clear is_valid_handle; % to clear init_key
dbstop if error

% a = gpuArray(1);clear a;
% caffe.reset_all();

run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% FillAmbigRegion('../carback_batch1/');

% do validation, or not
opts.do_val                 = true;
% model
model                       = Model.ZF_for_Faster_RCNN_VOC0712;
% model                       = Model.ZF_for_Faster_RCNN_CarBack;
% cache base
cache_base_proposal         = 'faster_rcnn_carback_ZF';
cache_base_fast_rcnn        = '';

%% ----------------- Load Data ----------------
carback_data = load('/home/hanxi/work/drivingEye_carback/JPEGImages_All/carback_voc_data.mat');
% dataset.
dataset = carback_data.fake_data;
dataset.roidb_train{1} = ModifyGT(dataset.roidb_train{1});
dataset.roidb_train{2} = ModifyGT(dataset.roidb_train{2});
dataset.roidb_test = ModifyGT(dataset.roidb_test);

%% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config('image_means', model.mean_image, 'feat_stride', model.feat_stride);
conf_fast_rcnn              = fast_rcnn_config('image_means', model.mean_image);
% set cache folder for each stage
model                       = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, cache_base_fast_rcnn, model);
% generate anchors and pre-calculate output size of rpn network
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);

%%  stage one proposal
fprintf('\n***************\nstage one proposal \n***************\n');
% train
model.stage1_rpn            = Faster_RCNN_Train.do_proposal_train(conf_proposal, dataset, model.stage1_rpn, opts.do_val);
% test
dataset.roidb_train        	= cellfun(@(x, y) Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage1_rpn, x, y), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
% dataset.roidb_test        	= Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);
dataset.roidb_train{1}.rois = MakeLogical(dataset.roidb_train{1}.rois);
dataset.roidb_train{2}.rois = MakeLogical(dataset.roidb_train{2}.rois);
dataset.roidb_test.rois = MakeLogical(dataset.roidb_test.rois);
save('faster_rcnn_lessclass_dataset_stage1.mat', 'dataset', '-v7.3');
% load('faster_rcnn_lessclass_dataset_stage1.mat');

%%  stage one fast rcnn
fprintf('\n***************\nstage one fast rcnn\n***************\n');
% train
model.stage1_fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train(conf_fast_rcnn, dataset, model.stage1_fast_rcnn, opts.do_val);
% test
% opts.mAP                    = Faster_RCNN_Train.do_fast_rcnn_test(conf_fast_rcnn, model.stage1_fast_rcnn, dataset.imdb_test, dataset.roidb_test);

%%  stage two proposal
% net proposal
fprintf('\n***************\nstage two proposal\n***************\n');
% train
model.stage2_rpn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_rpn            = Faster_RCNN_Train.do_proposal_train(conf_proposal, dataset, model.stage2_rpn, opts.do_val);
% test
dataset.roidb_train       	= cellfun(@(x, y) Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, x, y), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
% dataset.roidb_test       	= Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);
save('faster_rcnn_lessclass_dataset_stage2.mat', 'dataset', '-v7.3');
% load('faster_rcnn_lessclass_dataset_stage2.mat');

%%  stage two fast rcnn
fprintf('\n***************\nstage two fast rcnn\n***************\n');
% train
model.stage2_fast_rcnn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train(conf_fast_rcnn, dataset, model.stage2_fast_rcnn, opts.do_val);

%% final test
fprintf('\n***************\nfinal test\n***************\n');

model.stage2_rpn.nms        = model.final_test.nms;
% dataset.roidb_test       	= Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);
% opts.final_mAP              = Faster_RCNN_Train.do_fast_rcnn_test(conf_fast_rcnn, model.stage2_fast_rcnn, dataset.imdb_test, dataset.roidb_test);

% save final models, for outside tester
Faster_RCNN_Train.gather_rpn_fast_rcnn_models(conf_proposal, conf_fast_rcnn, model, dataset);
end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    [output_width_map, output_height_map] ...
                                = proposal_calc_output_size(conf, test_net_def_file);
    anchors                = proposal_generate_anchors(cache_name, ...
                                    'scales',  2.^[3:5]);
end
