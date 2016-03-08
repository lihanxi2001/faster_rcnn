% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Saturday 30/01/2016 03:01.
% Last Revision: Saturday 30/01/2016 03:01.

function [ap_all, prec_all, rec_all] = ...
    DetectionResEval(db_name, num_img, classes, thresh, ...
                class_pick, sameclass_nums_thres, res_folder)

    thresh_ap = 0.2;
    minoverlap = 0.5;
    A = cell2mat(cellfun(@(x) strcmpi(x, classes(:)'), ...
                        class_pick, 'UniformOutput', 0));
    class_idx = find(any(A, 1));

    rec_all = cell(1, length(classes));
    prec_all = cell(1, length(classes));
    ap_all =  zeros(1, length(classes));

    res_path = [res_folder, db_name, '_DetectRes_', 'ImgNum', num2str(num_img), ...
        '_Thresh', num2str(round(thresh * 100)), '_SameClassOverlap', ...
                num2str(round(sameclass_nums_thres * 10)), '.mat'];
    load(res_path) % aboxes_all, gt_all, img_name_all, classes

    for i_class = class_idx % 1 : num_classes

        gt_bbox = gt_all{i_class};
        gt_detected = cell(1, num_img);
        npos = 0;
        for i_img = 1 : num_img
            gt_detected{i_img} = false(1, size(gt_bbox{i_img}, 1));
            npos = npos + size(gt_bbox{i_img}, 1);
        end
        aboxes = aboxes_all{i_class};
        gtids = 1 : num_img;
        ids = bbox_ids(num_img, aboxes);
        aboxes = cell2mat(aboxes(:));
        small_idx = aboxes(:, 5) < thresh_ap;
        aboxes(small_idx, :) = [];
        ids(small_idx) = [];

        confidence = aboxes(:, 5);
        BB = aboxes(:, 1 : 4)';

        % sort detections by decreasing confidence
        [~, si] = sort(-confidence);
        ids = ids(si);
        BB = BB(:, si);

        % assign detections to ground truth objects
        cls = classes{i_class};
        nd = length(confidence);
        tp = zeros(nd, 1);
        fp = zeros(nd,1);

        tic;
        for d = 1 : nd
            % display progress
            if toc>1
                fprintf('%s: pr: compute: %d/%d\n', cls, d, nd);
                drawnow;
                tic;
            end

            % find ground truth image
            i_img = find(gtids == ids(d));
            if isempty(i_img)
                error('unrecognized image "%s"',ids{d});
            elseif length(i_img)>1
                error('multiple image "%s"',ids{d});
            end

            % assign detection to ground truth object if any
            bb = BB(:,d);
            ovmax = -inf;
            for i_bbox = 1 : size(gt_bbox{i_img}, 1)
                bbgt = gt_bbox{i_img}(i_bbox, :)';
                bi = [max(bb(1), bbgt(1)); max(bb(2), bbgt(2)); ...
                                min(bb(3), bbgt(3)); min(bb(4), bbgt(4))];
                iw = bi(3) - bi(1) + 1;
                ih = bi(4) - bi(2) + 1;
                if iw > 0 && ih > 0
                    % compute overlap as area of intersection / area of union
                    ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
                       (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
                       iw*ih;
                    ov=iw*ih/ua;
                    if ov>ovmax
                        ovmax=ov;
                        jmax=i_bbox;
                    end
                end
            end
            % assign detection as true positive/don't care/false positive
            if ovmax >= minoverlap
                if ~gt_detected{i_img}(jmax)
                    tp(d)=1;            % true positive
                    gt_detected{i_img}(jmax)=true;
                else
                    fp(d)=1;            % false positive (multiple detection)
                end
            else
                fp(d)=1;                    % false positive
            end
        end

        % compute precision/recall
        fp=cumsum(fp);
        tp=cumsum(tp);
        rec=tp/npos;
        prec=tp./(fp+tp);

        % compute average precision
        ap = 0;
        for t = 0 : 0.1 : 1
            p=max(prec(rec>=t));
            if isempty(p), p=0; end
            ap=ap+p/11;
        end

        % plot precision/recall
        plot(rec,prec,'-'); grid;
        xlabel 'recall'; ylabel 'precision';
        title(sprintf('class: %s, AP = %.3f', cls, ap));
        img_path = [res_folder, db_name, '_DetectRes_', 'ImgNum', num2str(num_img), ...
            '_Thresh', num2str(round(thresh * 100)), '_SameClassOverlap', ...
                num2str(round(sameclass_nums_thres * 10)), '_Obj', classes{i_class}, '_AP.png'];
        saveas(gcf, img_path);pause(0.2);clf;

        rec_all{i_class} = rec;
        prec_all{i_class} = prec;
        ap_all(i_class) = ap;

        fprintf('\nFor %s with overlap r = %2.2f, AP = %2.3f.\n', ...
                        classes{i_class}, sameclass_nums_thres, ap);

    end

    ap_path = [res_folder, db_name, '_DetectRes_', 'ImgNum', num2str(num_img), ...
                    '_Thresh', num2str(round(thresh * 100)), '_AP.mat'];
    save(ap_path, 'ap_all', 'prec_all', 'rec_all', 'classes', '-v7.3');

end

%% ****************************************************************** %%
function ids = bbox_ids(num_img, bbox_all)
    ids = cell(1, num_img);
    for i_img = 1 : num_img
        n_bbox = size(bbox_all{i_img}, 1);
        ids{i_img} = i_img * ones(n_bbox, 1);
    end
    ids = cell2mat(ids(:));
end

