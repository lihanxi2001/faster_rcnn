% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Monday 01/02/2016 00:54.
% Last Revision: Monday 01/02/2016 00:54.

clc;
close all;
clear;
dbstop if error

db_name = 'carback';
main_folder = '/home/hanxi/work/drivingEye_carback/test/';
save_folder_sub = '/JPEGImages_Test/';

FillAmbigRegion(db_name, main_folder, save_folder_sub);

load([main_folder, 'voc_data.mat']);
obj_names = {'Car'};
FakeVOCDataset(db_name, main_folder, save_folder_sub, voc_data, obj_names);




