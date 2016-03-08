function dataset = Conver20To1(src, dst, ncls)

carback_data = load(src);
dataset = carback_data.fake_data;
% dataset.roidb_train{1} = ModifyGT(dataset.roidb_train{1});
% dataset.roidb_train{2} = ModifyGT(dataset.roidb_train{2});
% dataset.roidb_test = ModifyGT(dataset.roidb_test);

dataset.imdb_train{1} = ConverImdb(dataset.imdb_train{1});
dataset.imdb_train{2} = ConverImdb(dataset.imdb_train{2});
dataset.imdb_test  = ConverImdb(dataset.imdb_test);

dataset.roidb_train{1} = ConverRoidb(dataset.roidb_train{1});
dataset.roidb_train{2} = ConverRoidb(dataset.roidb_train{2});
dataset.roidb_test  = ConverRoidb(dataset.roidb_test);

% dataset.roidb_train{1} = ModifyGT(dataset.roidb_train{1});
% dataset.roidb_train{2} = ModifyGT(dataset.roidb_train{2});
% dataset.roidb_test = ModifyGT(dataset.roidb_test);

save(dst, 'dataset', '-v7.3');

end

function dst = ConverImdb(src)

dst = src;
dst.classes = {'car'};
dst.num_classes = 1;
dst.class_ids = double(1);
dst.class_to_id = containers.Map('car',1);
dst.details.VOCopts.classes = {'car'};
dst.details.VOCopts.nclasses = 1;

end

function dst = ConverRoidb(src)

dst = src;
for i = 1 : length(dst.rois)
    dst.rois(i).class = dst.rois(i).class / 7;
    dst.rois(i).overlap = dst.rois(i).overlap(:,7);
end

end