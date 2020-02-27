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
%images = db.dbImageFns;
images = db.qImageFns;
images_paths = '/mnt/1E48BE700AFD16C7/datasets/247dataset/247_Tokyo_GSV/query/';


%% EDGE BOX
%load pre-trained edge detection model and set opts (see edgesDemo.m)
boxes_path = '/mnt/1E48BE700AFD16C7/datasets/247dataset/247_Tokyo_GSV/vt/q_boxes_0_315.mat';

boxx = load(boxes_path);

%% START
load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );
net= relja_simplenn_tidy(net);

for i = 1:size(images)
        file_name = strcat(images_paths,images(i)); 
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

        if rem(i,500) == 0
            j = i-500;
            filemat_name = strcat(Save_path,'db_feats_',num2str(j),'_',num2str(i),'.mat');
            save(char(filemat_name),'feats_file');
            
         
            
            fileID = fopen('status-leocomputerrepresentation.txt','w');
            fprintf( '==>> %i/%i ~ %% %f ',i,length(images), i/length(images)*100);
            fclose(fileID);
            
            clear feats_file;
            clear bbox_file;
        end
        
        
    
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
j = 0; 
filemat_name = strcat(Save_path,'db_feats_',num2str(j),'_',num2str(i),'.mat');
save(char(filemat_name),'feats_file');
            