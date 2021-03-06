
% run /mnt/0287D1936157598A/docker_ws/docker_ws/netvlad/netvlad-orignal/leo_computeRepresentation_box_datasets.m

clear all;
clc;



%% load pre-trained edge detection model and set opts (see edgesDemo.m)

model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .85;     % step size of sliding window search0.65
opts.beta  = .8;     % nms threshold for object proposals0.75
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 200;  % max number of boxes to detect 1e4

%%

addpath(genpath('/home/leo/docker_ws/netvlad'));

setup;


Dataset_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV/dataset/query';

addpath(genpath(Dataset_path));

Save_path_1_e ='/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV/vt';


paths= localPaths();

%%

db = dbTokyo247();

images_paths = paths.dsetRootTokyo247;

%images = db.dbImageFns;
images = db.qImageFns;

netID= 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white';

% make two bin files for all

% 1-3101
% 1.mat- 3101-3601
% afterwards all files are of hundreds

load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );
net= relja_simplenn_tidy(net);
%j = 3601;
for i = 1:size(images)
        file_name = strcat(images_paths,images(i)); 
        im= vl_imreadjpeg({char(file_name)}); 
        I = uint8(im{1,1});
        [bboxes, E] =edgeBoxes(I,model);
        results = uint8(E * 255);
        bboxes_1000 = bboxes;
        






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




all_images = load('/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV/qTokyoTMImageFns.mat');
images = all_images.aa{1,1};




tElapsed = zeros(10,1);
time_i = 1;

bboxes_1000 = struct;
jj = 0;

for i = 1:size(images)
          
    file_name = strcat(Dataset_path,"/",images(i)); 
    save_edge = strcat(Save_path_1_e,"/",images(i)); 

      tStart = tic; 
  
    %% Read and Process Image
    
    
    
    I = imread(char(file_name));
    [bboxes, E] =edgeBoxes(I,model);
    results = uint8(E * 255);
    bboxes_1000 = bboxes;
    %if i-jj == 5
   % filemat_name = strcat('/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV/','q','_',jj,'_',i,'.mat');
   % save(filemat_name,'bboxes_1000');
   % bboxes_1000 = [];

    %end    
%% Create View Tags
    
% Create Edge VT
   e8u_norml_values = norml_values_strict(E,1,0,1);     % (xx,max_value,min1,max1)

   e8u_c1 = e8u_norml_values;
   e8u_c1(e8u_c1>.33)  = 0;
   e8u_c1 = norml_values_strict(e8u_c1,1,0,0.33);

   e8u_c2 = e8u_norml_values;
   e8u_c2(e8u_c2<.33)  = 0;
   e8u_c2(e8u_c2>.66)  = 0;
   e8u_c2 = norml_values_strict(e8u_c2,1,0.33,0.66);

   e8u_c3 = e8u_norml_values;
   e8u_c3(e8u_c3<.66)  = 0;
   e8u_c3 = norml_values_strict(e8u_c3,1,.66,1);

   c1_mat_255 = uint8(e8u_c1* 255);
   c2_mat_255 = uint8(e8u_c2* 255);
   c3_mat_255 = uint8(e8u_c3* 255);
    
    

   mat_255 = c1_mat_255+c2_mat_255+c3_mat_255;
   
   [folder, baseFileName, extension] = fileparts(char(save_edge));
        if exist(folder, 'dir')==0
            mkdir(char(folder))
   end
    
   imwrite(mat_255, char(save_edge))

  %  % Saving method
   
   %    bboxes_1000(1:50,((i-1*5)+1):i*5) = bboxes(1:50,1:5);

  
   
    query_display = sprintf('%d / %d',i,length(images));
    disp(query_display)

end

filemat_name = strcat('/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV/','q','_all.mat');
save(filemat_name,'bboxes_1000');
bboxes_1000 = [];


function new_filePath = create_filepath_file(Parent_dir, file_dir, file_name)
new_filePath = strcat(Parent_dir,file_dir); %filepath is between savepath (might need to create the directories)
     %% Save the original edge image
    if exist(new_filePath, 'dir')==0
      mkdir(char(new_filePath))
    end
    new_filePath = strcat(new_filePath,"/",file_name);
end

function y = norml_values_strict(xx,max_value,min1,max1)
    y=((xx-min1).*max_value)./(max1-min1);
end
