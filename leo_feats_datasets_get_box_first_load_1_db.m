% run /mnt/0287D1936157598A/docker_ws/docker_ws/netvlad/netvlad-orignal/leo_computeRepresentation_box_datasets.m

clear all;
clc;
setup;
paths= localPaths();

%% DATAPATH

%%XPS
addpath(genpath('/mnt/0287D1936157598A/docker_ws/docker_ws/netvlad/netvlad-orignal'));


%% DATASET

% PITTSBURG
%netID= 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white'; % its in the CNN
%db= dbPitts('30k', 'test');
%images = db.dbImageFns;
%images = db.qImageFns;
%images_paths = '/home/leo/docker_ws/datasets/Pittsburgh-all/Pittsburgh/queries/';

% TOKYO247
%Save_path ='/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV/vt/';
Save_path ='/mnt/1E48BE700AFD16C7/datasets/247dataset/247_Tokyo_GSV/vt/';
netID= 'vd16_tokyoTM_conv5_3_vlad_preL2_intra_white'; % its in the CNN
db = dbTokyo247();
images = db.dbImageFns;
images_paths = '/mnt/1E48BE700AFD16C7/datasets/247dataset/247_Tokyo_GSV/images/';

%images = db.qImageFns;




%% START
load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );
net= relja_simplenn_tidy(net);

batch_start = 0; 

for jj = 1:17
%% EDGE BOX
%load pre-trained edge detection model and set opts (see edgesDemo.m)
batch_last = jj*5000;
boxes_path = '/mnt/1E48BE700AFD16C7/datasets/247dataset/247_Tokyo_GSV/vt/db_boxes_0_5000.mat';
filemat_name = strcat(Save_path,'db_boxes_',num2str(batch_start),'_',num2str(batch_last),'.mat');

boxx = load(boxes_path);

    for i = 1:5000
            file_name = strcat(images_paths,images(i+batch_start)); 
            im= vl_imreadjpeg({char(file_name)}, 'numThreads', 12); 
            
            bboxes = boxx.bbox_file(i).testq;
            %bboxes = bob(i).testq;
            % to preserve the spatial information
            mat_boxes = uint8(bboxes/16); 
            [wd, hh] = size(im{1,1});
            %size(mat_boxes) (if boxes are less then 50 -> create empty boxes
            while (size(mat_boxes) < 50)
                mat_boxes_add = [0 0 480/16-1 hh/16-1 0]; 
                mat_boxes( end+1, : ) = mat_boxes_add; 
                size(mat_boxes)

            end
            
            
            im= im{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
            feats= leo_computeRepresentation(net, im, mat_boxes); % add `'useGPU', false` if you want to use the CPU
        
            feats_file(i) = struct ('featsdb', feats); 

            clear feats;
            clear im;
            clear aq;
            clear mat_boxes;
            clear filemat_name; 
            clear file_name;
            clear mat_name;
            clear Mat_file;
            

            fprintf( '  ==>> %i/%i ',i,length(images));



    end

    
    filemat_name = strcat(Save_path,'db_feats_',num2str(batch_start),'_',num2str(i+batch_start),'.mat');
    save(char(filemat_name),'feats_file');
    batch_start = batch_start + 5000; 

end     