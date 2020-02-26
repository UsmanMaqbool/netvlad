% run /mnt/0287D1936157598A/docker_ws/docker_ws/netvlad/netvlad-orignal/leo_computeRepresentation_box_datasets.m

clear all;
clc;
setup;
paths= localPaths();

%% DATAPATH

%%XPS
addpath(genpath('/home/leo/docker_ws/netvlad/SELN-0.1-box'));

Save_path ='/home/leo/docker_ws/datasets/Pittsburgh-all/vt/';

%% DATASET

% PITTSBURG
netID= 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white'; % its in the CNN

db= dbPitts('30k', 'test');
%images = db.dbImageFns;
images = db.qImageFns;
images_paths = '/home/leo/docker_ws/datasets/Pittsburgh-all/Pittsburgh/queries/';

% TOKYO247
%db = dbTokyo247();
%images_paths = paths.dsetRootTokyo247;
%images = db.dbImageFns;
%images = db.qImageFns;
%netID= 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white';
%images_paths = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV/dataset/query';


%% EDGE BOX
%load pre-trained edge detection model and set opts (see edgesDemo.m)

model=load('edges/models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .85;     % step size of sliding window search0.65
opts.beta  = .8;     % nms threshold for object proposals0.75
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 200;  % max number of boxes to detect 1e4
gt=[111	98	25	101];
opts.minBoxArea = 0.5*gt(3)*gt(4);
opts.maxAspectRatio = 1.0*max(gt(3)/gt(4),gt(4)./gt(3));

%% START
load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );
net= relja_simplenn_tidy(net);
j = 0;
for i = 1:size(images)
        file_name = strcat(images_paths,images(i)); 
        im= vl_imreadjpeg({char(file_name)}); 
        I = uint8(im{1,1});
        [bbx, E] =edgeBoxes(I,model);
        %results = uint8(E * 255); if you want to save the images -> then
        %use it
        
        %make empty list of small boxes 
        bboxes=[]; 
        b_size = size(bbx,1); 
        for ii=1:b_size
             bb=bbx(ii,:);
             square = bb(3)*bb(4);
             if square <2*gt(3)*gt(4)
                bboxes=[bbx;bb];
             end
        end
        
        bbox_file(i) = struct ('testdb', bboxes); 
        
        
        % to preserve the spatial information
        mat_boxes = uint8(bboxes/16); 
        
        %size(mat_boxes) (if boxes are less then 50 -> create empty boxes
        while (size(mat_boxes) < 50)
            mat_boxes_add = [0 0 30 40 0]; 
            mat_boxes( end+1, : ) = mat_boxes_add; 
            size(mat_boxes)

        end
        
        
        im= im{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
        feats= leo_computeRepresentation(net, im, mat_boxes); % add `'useGPU', false` if you want to use the CPU
    
        feats_file(i) = struct ('featsdb', feats); 

        if i - j == 5
            filemat_name = strcat(Save_path,'db_feats_',num2str(j),'_',num2str(i),'.mat');
            save(char(filemat_name),'feats_file');
            
            filemat_name = strcat(Save_path,'db_boxes_',num2str(j),'_',num2str(i),'.mat');
            save(char(filemat_name),'bbox_file');
            
            j = i;
            
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

