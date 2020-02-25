
% run /mnt/0287D1936157598A/docker_ws/docker_ws/netvlad/netvlad-orignal/leo_computeRepresentation_box_datasets.m

clear all;
clc;

addpath(genpath('/mnt/0287D1936157598A/docker_ws/docker_ws/netvlad/leo-netvlad'));

setup;

mat_paths = '/mnt/1E48BE700AFD16C7/datasets/Pittsburgh_Viewtag_1_mat';

paths= localPaths();

db= dbPitts('30k', 'test');
images = db.dbImageFns;
%images = db.qImageFns;

netID= 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white';

% make two bin files for all

% 1-3101
% 1.mat- 3101-3601
% afterwards all files are of hundreds

load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );
net= relja_simplenn_tidy(net);
%j = 3601;
for i = 1:size(images)
    add_missing_file = '000/000414_pitch1_yaw10.jpg';

    if strcmp(images(i), add_missing_file) == 1
        file_name = strcat(paths.dsetRootPitts,"/",images(i)); 
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
    end
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
        file_name = strcat(paths.dsetRootPitts,"/",images(i)); 
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