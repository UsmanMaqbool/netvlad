% run /mnt/0287D1936157598A/docker_ws/docker_ws/netvlad/netvlad-orignal/leo_computeRepresentation_box_datasets.m

clear all;
clc;
setup;
paths= localPaths();

%% DATAPATH

%%XPS
addpath(genpath('/home/leo/docker_ws/netvlad/SELN-0.1-box'));

Save_path_1_e ='/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV/vt';

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
%j = 3601;
for i = 1:size(images)
        file_name = strcat(images_paths,images(i)); 
        im= vl_imreadjpeg({char(file_name)}); 
        I = uint8(im{1,1});
        [bbx, E] =edgeBoxes(I,model);
        results = uint8(E * 255);
        
         
        bboxes=[]; %make empty list of small boxes
        b_size = size(bbx,1); 
        for ii=1:b_size
             bb=bbx(ii,:);
             square = bb(3)*bb(4);
             if square <2*gt(3)*gt(4)
                bboxes=[bbs1;bb];
             end
        end
        
        bbox_file(i) = struct ('testdb', bboxes); 
        %% SAVE
        
        
        
        mat_boxes = uint8(bboxes/16); % to preserve the spatial information
        %size(mat_boxes)
        while (size(mat_boxes) < 50)
            mat_boxes_add = [0 0 30 40 0]; 
            mat_boxes( end+1, : ) = mat_boxes_add; 
            size(mat_boxes)

        end
        
        


        
        im= im{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
        feats= leo_computeRepresentation(net, im, mat_boxes); % add `'useGPU', false` if you want to use the CPU
    
        filemat_name = strcat('/mnt/1E48BE700AFD16C7/datasets/output-files/db','/',mat_name);
        [folder, baseFileName, extension] = fileparts(char(filemat_name));
        if exist(folder, 'dir')==0
            mkdir(char(folder))
        end

        save(char(filemat_name),'feats');
    
        clear feats;
        clear im;
        clear aq;
        clear mat_boxes;
        clear filemat_name; 
        clear file_name;
        clear mat_name;
        clear Mat_file;
        %res_b{i} = feats;
    %query_display = sprintf( '%i/%i ~ %% %f ',i,length(images), i/length(images)*100);
    %cd disp(query_display)
        
        fileID = fopen('status-leocomputerrepresentation.txt','w');
        fprintf( '==>> %i/%i ~ %% %f ',i,length(images), i/length(images)*100);
        fclose(fileID);

%   #if (i-j) == 500
%
 %       j = i;
  %      res_b = [];
   %     res_b = { struct('feat', cell(4096,50))}; 
   % end 
end

clear images;
images = db.qImageFns;
for i = 1:size(images)
    add_missing_file = '000/000414_pitch1_yaw6.jpg';
    if images(i) == char(add_missing_file )
        file_name = strcat(paths.dsetRootTokyo247,"/",images(i)); 
        mat_name = strrep(images(i),'.jpg','.mat');
        Mat_file = strcat(mat_paths,"/",mat_name); 
        aq = load(Mat_file);

        im= vl_imreadjpeg({convertStringsToChars(file_name)}); 
        mat_boxes = uint8(aq.bboxes/16); % to preserve the spatial information
        %size(mat_boxes)
        while (size(mat_boxes) < 50)
            mat_boxes_add = [0 0 30 40 0]; 
            mat_boxes( end+1, : ) = mat_boxes_add; 
            size(mat_boxes)

        end
        im= im{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
        feats= leo_computeRepresentation(net, im, mat_boxes); % add `'useGPU', false` if you want to use the CPU
    
        filemat_name = strcat('/mnt/1E48BE700AFD16C7/datasets/output-files/q','/',mat_name);
        [folder, baseFileName, extension] = fileparts(char(filemat_name));
        if exist(folder, 'dir')==0
            mkdir(char(folder))
        end

        save(char(filemat_name),'feats');
    
        clear feats;
        clear im;
        clear aq;
        clear mat_boxes;
        clear filemat_name; 
        clear file_name;
        clear mat_name;
        clear Mat_file;
        %res_b{i} = feats;
    %query_display = sprintf( '%i/%i ~ %% %f ',i,length(images), i/length(images)*100);
    %cd disp(query_display)
        
        fileID = fopen('status-leocomputerrepresentation.txt','w');
        fprintf( '==>> %i/%i ~ %% %f ',i,length(images), i/length(images)*100);
        fclose(fileID);
    end
%   #if (i-j) == 500
%
 %       j = i;
  %      res_b = [];
   %     res_b = { struct('feat', cell(4096,50))}; 
   % end 
end


