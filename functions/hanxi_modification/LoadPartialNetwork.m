% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Sunday 17/01/2016 17:17.
% Last Revision: Monday 18/01/2016 21:31.

function LoadPartialNetwork(full_model)

    caffe_solver = caffe.Solver(opts.solver_def_file);
    caffe_solver.net.copy_from(opts.net_file);

    rpn_net = caffe.Net(proposal_detection_model.proposal_net_def, 'test');
    rpn_net.copy_from(proposal_detection_model.proposal_net);

    rpn_net = caffe.Net(proposal_detection_model.proposal_net_def, 'test');
    rpn_net.copy_from(proposal_detection_model.proposal_net);

end



