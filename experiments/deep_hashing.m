% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Wednesday 16/12/2015 15:29.
% Last Revision: Wednesday 16/12/2015 15:29.

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% model
model                       = Model.ZF_for_Faster_RCNN_VOC0712;
% cache base
cache_base_proposal         = 'faster_rcnn_VOC0712_ZF';
cache_base_fast_rcnn        = '';
% train/test data
dataset                     = [];
use_flipped                 = true;
dataset                     = Dataset.voc0712_trainval(dataset, 'train', use_flipped);
dataset                     = Dataset.voc2007_test(dataset, 'test', false);



