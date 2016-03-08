% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Sunday 17/01/2016 17:17.
% Last Revision: Tuesday 19/01/2016 00:29.

function MakePartialNetwork()

    %% Cut the original large network and feed the partial parameters into
    % the small one.
    caffe.reset_all();

    % For win users, you need to modify the following line.
    main_path = '/home/hanxi/work/faster_rcnn-master_lessclass/';

    partial_solver_file = [main_path, '/models/fast_rcnn_prototxts/ZF_Half/solver_30k40k.prototxt'];
    partial_model_file = [main_path, '/models/pre_trained_models/ZF_Half/ZF_Half.caffemodel'];
    partial_solver = caffe.Solver(partial_solver_file);

    reduce_layer = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'};
    rpn_def_file = [main_path, '/output/faster_rcnn_final/faster_rcnn_VOC0712_ZF/proposal_test.prototxt'];
    rpn_model_file = [main_path, '/output/faster_rcnn_final/faster_rcnn_VOC0712_ZF/proposal_final'];
    rpn_net = caffe.Net(rpn_def_file, 'test');
    rpn_net.copy_from(rpn_model_file);
    for i_layer = 1 : length(reduce_layer)
        rpn_param = rpn_net.params(reduce_layer{i_layer}, 1).get_data();
        half_param = partial_solver.net.params(reduce_layer{i_layer}, 1).get_data();
        m_ch_in = size(half_param, 3);
        m_ch_out = size(half_param, 4);
        half_param = rpn_param(:, :, 1 : m_ch_in, 1 : m_ch_out);
        partial_solver.net.params(reduce_layer{i_layer}, 1).set_data(half_param);

        rpn_param = rpn_net.params(reduce_layer{i_layer}, 2).get_data();
        half_param = partial_solver.net.params(reduce_layer{i_layer}, 2).get_data();
        m_ch_in = size(half_param, 1);
        half_param = rpn_param(1 : m_ch_in, :);
        partial_solver.net.params(reduce_layer{i_layer}, 2).set_data(half_param);
    end

    reduce_layer = {'fc6', 'fc7'};
    det_def_file = [main_path, '/output/faster_rcnn_final/faster_rcnn_VOC0712_ZF/detection_test.prototxt'];
    det_model_file = [main_path, '/output/faster_rcnn_final/faster_rcnn_VOC0712_ZF/detection_final'];
    det_net = caffe.Net(det_def_file, 'test');
    det_net.copy_from(det_model_file);
    for i_layer = 1 : length(reduce_layer)
        det_param = det_net.params(reduce_layer{i_layer}, 1).get_data();
        half_param = partial_solver.net.params(reduce_layer{i_layer}, 1).get_data();
        m_ch_in = size(half_param, 1);
        m_ch_out = size(half_param, 2);
        half_param = det_param(1 : m_ch_in, 1 : m_ch_out);
        partial_solver.net.params(reduce_layer{i_layer}, 1).set_data(half_param);
    end

    partial_solver.net.save(partial_model_file);

    %% Test
    caffe_solver = caffe.Solver(partial_solver_file);
    caffe_solver.net.copy_from(partial_model_file);

end



