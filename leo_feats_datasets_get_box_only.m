% run /mnt/0287D1936157598A/docker_ws/docker_ws/netvlad/netvlad-orignal/leo_computeRepresentation_box_datasets.m

clear all;
clc;
setup;
paths= localPaths();

%% DATAPATH

%%Euler Cluster
addpath(genpath('/cluster/home/mbhutta/docker_ws/netvlad'));


%% DATASET

% PITTSBURG
netID= 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white'; % its in the CNN
db= dbPitts('30k', 'test');
images = db.dbImageFns;
images_paths = '/cluster/scratch/mbhutta/Test_Pitts30k/images/';

%images = db.qImageFns;
%images_paths = '/cluster/scratch/mbhutta/Test_Pitts30k/queries/';

Save_path ='/cluster/scratch/mbhutta/Test_Pitts30k/vt/';


% TOKYO247
%netID= 'vd16_tokyoTM_conv5_3_vlad_preL2_intra_white'; % its in the CNN
%db = dbTokyo247();
%images = db.dbImageFns;
%images_paths = '/cluster/scratch/mbhutta/Test_247_Tokyo_GSV/images/';

%images = db.qImageFns;
%images_paths = '/cluster/scratch/mbhutta/Test_247_Tokyo_GSV/query/';
%Save_path ='/cluster/scratch/mbhutta/Test_247_Tokyo_GSV/vt/';

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

for i = 1:size(images)
        file_name = strcat(images_paths,images(i)); 
        im= vl_imreadjpeg({char(file_name)},'numThreads', 12); 
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
        
        bbox_file(i) = struct ('testq', bboxes); 
        
                

        if rem(i,5000) == 0
            if i == 75984
                j == 75000;
            else
                j = i-5000;
            end
            
            filemat_name = strcat(Save_path,'db_boxes_',num2str(j),'_',num2str(i),'.mat');
            save(char(filemat_name),'bbox_file');
            
            fileID = fopen('status-leocomputerrepresentation.txt','w');
            fprintf( '==>> %i/%i ~ %% %f ',i,length(images), i/length(images)*100);
            fclose(fileID);
            
            clear feats_file;
            clear bbox_file;
        end
        
        
    
       
        clear im;
        clear aq;
        clear mat_boxes;
        clear filemat_name; 
        clear file_name;
        clear mat_name;
        clear Mat_file;
        

        fprintf( '  ==>> %i/%i ',i,length(images));



end
j = j+5000;


filemat_name = strcat(Save_path,'db_boxes_',num2str(j),'_',num2str(i),'.mat');
save(char(filemat_name),'bbox_file');