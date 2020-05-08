 function [res, recalls]= leo_recallAtN(searcher, nQueries, isPos, ns, printN, nSample,db,plen_opts)
    if nargin<6, nSample= inf; end
    
    rngState= rng;
    
    if nSample < nQueries
        rng(43);
        toTest= randsample(nQueries, nSample);
    else
        toTest= 1:nQueries;
    end
    
    assert(issorted(ns));
    nTop= max(ns);
    
    recalls= zeros(length(toTest), length(ns));
    recalls_ori= zeros(length(toTest), length(ns));
    printRecalls= zeros(length(toTest),1);
    
    evalProg= tic;
    %dataset_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV';
    %save_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV/vt-2';
   
    % detect blackish images  ( plot the boxes)
    % rearrange + use previous knowledge
    

    
    vt_type = plen_opts.vt_type;
    iTestSample_Start=plen_opts.iTestSample_Start; startfrom =plen_opts.startfrom; show_output = plen_opts.show_output;  %test the boxes
     dataset_path = plen_opts.dataset_path; 
     save_path = plen_opts.save_path; 
    %% LEO START
    
    netID= plen_opts.netID;
    % netID= 'caffe_tokyoTM_conv5_vlad_preL2_intra_white';


    paths= localPaths();

    load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );

    %%
    net= relja_simplenn_tidy(net); % potentially upgrate the network to the latest version of NetVLAD / MatConvNet

    
    
    
    
    %addpath(genpath('/mnt/02/docker_ws/docker_ws/netvlad/slen-0.2-box'));
    
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
    num_box = 9; % Total = 10 (first one is the full images feature / box)
    
    
    Top_boxes = 10;

    top_100 = [];
    total_top = 100; %100;
 
   
  
 

    for iTestSample= iTestSample_Start:length(toTest)
        
        %Display
        relja_progress(iTestSample, ...
                       length(toTest), ...
                       sprintf('%.4f', mean(printRecalls(1:(iTestSample-1)))), evalProg);
        
     
        iTest= toTest(iTestSample);
        
        [ids ds_pre]= searcher(iTest, nTop); % Main function to find top 100 candidaes
        ds_pre_max = max(ds_pre); ds_pre_min = min(ds_pre);
       
        %% Leo START
                
        qimg_path = strcat(dataset_path,'/queries/', db.qImageFns{iTestSample, 1});  
        q_img = strcat(save_path,'/', db.qImageFns{iTestSample, 1});  
        q_feat = strrep(q_img,'.jpg','.mat');
        
        if show_output == 1
        subplot(2,2,1); imshow(imread(char(qimg_path))); %q_img
        db_img = strcat(dataset_path,'/images/', db.dbImageFns{ids(1,1),1});  

        subplot(2,2,2); imshow(imread(char(db_img))); %
        hold;
        end
        
        if exist(q_feat, 'file')
            load(q_feat);
            if vt_type == 3
            imgg_mat_box_q = bbox;
            end
        else
            im= vl_imreadjpeg({char(qimg_path)},'numThreads', 12); 

           I = uint8(im{1,1});
            [bbox, E] =edgeBoxes(I,model);
            
           % [bbox,im, E, wd, hh] = img_Bbox(qimg_path,model);
            
            [wd, hh] = size(im{1,1});
            mat_boxes = leo_slen_increase_boxes(bbox,wd,hh);

            im= im{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
            query_full_feat= leo_computeRepresentation(net, im, mat_boxes,num_box); % add `'useGPU', false` if you want to use the CPU
            
            if vt_type == 2
            save(q_feat,'query_full_feat');
 
            end
            
            if vt_type == 3
                check_folder = fileparts(q_feat);
                if ~exist(check_folder, 'dir')
                   mkdir(check_folder)
                end
            save(q_feat,'query_full_feat', 'bbox', 'E');
            end
        end
      
        q_dbfeat = strrep(q_feat,'.mat','_db_feats.mat');
        if exist(q_dbfeat, 'file')
            if vt_type == 2
                    load(q_dbfeat); 
            end
            
            if vt_type == 3
                    q_dbfeat_all = load(q_dbfeat);
            end

        else
            % Top 100 sample
            for jj = 1:total_top

                    db_img = strcat(dataset_path,'/images/', db.dbImageFns{ids(jj,1),1});  
                    im= vl_imreadjpeg({char(db_img)},'numThreads', 12); 
                    I = uint8(im{1,1});
                    [bbox, E] =edgeBoxes(I,model);
                     [wd, hh] = size(im{1,1});   % update the size accordign to the DB images. as images have different sizes. 
                  %  [bbox,im, E, wd, hh] = img_Bbox(db_img,model);
                   
                    mat_boxes = leo_slen_increase_boxes(bbox,wd,hh);
                    
                    im= im{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
                    feats= leo_computeRepresentation(net, im, mat_boxes,num_box); % add `'useGPU', false` if you want to use the CPU
                    feats_file(jj) = struct ('featsdb', feats); 
                    bbox_file(jj) = struct ('bboxdb', bbox); 
                    Edge_images(jj) = struct ('Edges', E); 
                    clear feats;
                    fprintf( '==>> %i ~ %i/%i ',iTestSample,jj,total_top );

            end
            
            if vt_type == 2
                  save(q_dbfeat,'feats_file');

            end
            
            if vt_type == 3
                check_folder = fileparts(q_dbfeat);
                if ~exist(check_folder, 'dir')
                   mkdir(check_folder)
                end
                save(q_dbfeat,'feats_file','bbox_file', 'Edge_images');
                q_dbfeat_all = load(q_dbfeat);

            end
        end
        SLEN_top = zeros(total_top,2); 
        k = Top_boxes;
        ds_all = [];
       % figure;

        for i=startfrom:total_top 
            if vt_type == 2
             feats2 = feats_file(i).featsdb;

            end
            
            if vt_type == 3
            feats2 = q_dbfeat_all.feats_file(i).featsdb;
            imgg_mat_box_db = q_dbfeat_all.bbox_file(i).bboxdb;
            end
            for j = 1:Top_boxes
                q1 = single(feats2(:,j));  %take column of each box
                [ids1, ds1]= yael_nn(query_full_feat, q1, k);
                ds_all = [ds_all ds1];
            end
            % original dis: 1.25 ds_pre
            db_img = strcat(dataset_path,'/images/', db.dbImageFns{ids(i,1),1});  
            
            ds_pre_inv = 1/ ds_pre(i,1);
            
            ds_all_inv = 1./(ds_all);
            
            diff_ds_all = zeros(Top_boxes,Top_boxes);
            diff_ds_all(1:Top_boxes-1,:) = diff(ds_all);
            diff2_ds_all = diff(diff(ds_all));
            diff2_ds_all_less = diff2_ds_all;
            diff2_ds_all_less(diff2_ds_all_less>0) = 0;
            diff_ds_all_inv = diff(ds_all_inv);
        
            
            ds_all_sub = ds_all(1:Top_boxes,1:Top_boxes);
            ds_all_sub_inv = ds_all_inv(1:Top_boxes,1:Top_boxes);
            
            
            ds_all_less = ds_all_sub-ds_pre(100,1);
            ds_all_less_inv = ds_all_sub_inv-ds_pre_inv;
            
            
            ds_all_less_mean = mean(ds_all_less(:));
            ds_all_less_inv_mean = mean(ds_all_less_inv(:));
            
            s=sign(ds_all_less); s_inv=sign(ds_all_less);
            
            ipositif=sum(s(:)==1);
            inegatif=sum(s(:)==-1);
            S_great = s; S_great(S_great<0) = 0; S_great = S_great.*ds_all_less; S_great_n = S_great - ds_all_less_mean;
            S_less = s; S_less(S_less>0) = 0; S_less = abs(S_less).*ds_all_less; S_less_n = S_less - ds_all_less_mean;
            S_less_diff = diff(S_less);
            
            ipositif_inv=sum(s_inv(:)==1);
            inegatif_inv=sum(s_inv(:)==-1);
            S_great_inv = s_inv; S_great_inv(S_great_inv<0) = 0; S_great_inv = S_great_inv.*ds_all_less_inv; S_great_inv_n = S_great_inv - ds_all_less_inv_mean; 
            S_less_inv = s_inv; S_less_inv(S_less_inv>0) = 0; S_less_inv = abs(S_less_inv).*ds_all_less_inv; S_less_inv_n = S_less_inv - ds_all_less_inv_mean;

            %  [S_less_min_inv, S_less_I_inv] = sort(S_less_inv(:));
            

           S_great_mean = sum(S_great(:)/ipositif); S_great_n_mean = sum(S_great_n(:)/ipositif);
           S_great_inv_mean = sum(S_great_inv(:)/ipositif_inv); S_great_inv_n_mean = sum(S_great_inv_n(:)/ipositif_inv);
           
           
           
           
           S_less_mean = sum(sum(S_less/inegatif)); S_less_n_mean = sum(S_less_n(:)/inegatif);
           S_less_inv_mean = sum(S_less_inv(:)/inegatif_inv); S_less_inv_n_mean = sum(S_less_inv_n(:)/inegatif_inv);
          
             
           S_less_diff = diff(S_less); 
           S_less_n_diff = sum(sum(S_less_n.*diff_ds_all));
           S_less_inv_diff = sum(sum(S_less_inv.*diff_ds_all)); S_less_inv_n_diff = sum(sum(S_less_inv_n.*diff_ds_all));
          
           
           
          %  subplot(2,2,1); h = heatmap(S_less);
           % subplot(2,2,2); h = heatmap(S_less_n);
           % subplot(2,2,3); h = heatmap(S_less.*diff_ds_all);
           % subplot(2,2,4); h = heatmap(S_less_n.*diff_ds_all);
            
           % subplot(2,2,3); h = heatmap(S_less_inv);
           % subplot(2,2,4); h = heatmap(S_less_inv_n);
            
           
           
            s_delta_all = 0;
    
            
            s_delta_mat = 0;
            s_dis = 0;
            for jj = 1:Top_boxes
                S_less_col = S_less(:,jj);
                s_near_mat = [];
                for jjj = 1:Top_boxes-1
                                           
                        s_dis = abs(ds_pre(i,1) - S_less_col(jjj));
                        s_less_difference = abs(S_less_col(jjj+1)-S_less_col(jjj));
                        if s_less_difference > 0 && s_less_difference <= 0.02 && s_dis > .4
                            if isempty(s_near_mat)
                                
                                
                                s_near_mat = [s_near_mat; S_less_col(jjj);S_less_col(jjj+1)];
                                
                            else
                                s_near_mat = [s_near_mat; S_less_col(jjj+1)];
                            end
                            s_delta = exp(s_dis)*(s_less_difference)^jj;
                            s_delta_mat = s_delta_mat + s_delta;
                        elseif s_less_difference > 0 && s_less_difference > 0.03
                           % s_delta_mat = [s_delta_mat s_near_mat];
                            s_near_mat = [];
                        end
                    
                   
                end
               % s_delta_mat = [s_delta_mat s_near_mat];
                s_near_mat = [];
            end
                   
            
            D_diff = ds_pre(i,1); %-s_delta_all;
            
            
            D = sum(sum(S_less(1:Top_boxes)));
            boxes_per_less = (Top_boxes*Top_boxes)/inegatif;
            % Create plots
            if show_output == 2

                subplot(2,2,1); imshow(imread(char(qimg_path))); %q_img
                subplot(2,2,2); imshow(imread(char(db_img))); %

                
                subplot(2,2,3); h = heatmap(diff2_ds_all_less*ds_pre_inv);
                subplot(2,2,4); h = heatmap(diff2_ds_all*ds_pre_inv); % with plus is wokring
              %  subplot(2,2,1); h = heatmap(diff_ds_all/ds_pre_inv);
              %  subplot(2,2,2); h = heatmap(diff2_ds_all/ds_pre_inv);
%                fprintf( '==>> Distance %f ~ Greator Values %f %f \n Less Values %f %f ~ Min %f \n',ds_pre(i,1), s_delta_all,ipositif, S_great, inegatif, S_less);
              %  fprintf('%f %f %f %f %f %f %f %f %f %f\n',(Top_boxes*Top_boxes)/inegatif, D_diff, D_diff+ds_all_less_mean, D_diff-(ds_all_less_mean/D), D_diff-s_delta_mat, D_diff+s_delta_mat,ds_all_less_mean+s_delta_mat, s_delta_mat/ds_all_less_mean, D/(D_diff-ds_all_less_mean), D);
               % fprintf('\n For %f %f %f %f %f %f %f %f %f', D_diff, D, boxes_per_less, Top_boxes, inegatif, s_delta_mat, ds_all_less_mean, mean(S_less_diff(:)),min(S_less_diff(:)));
                fprintf('%f -> %f %f %f %f %f %f %f %f \n',D_diff,D_diff+ S_less_mean,D_diff+ S_less_n_mean, D_diff+S_less_inv_mean,D_diff+ S_less_inv_n_mean,D_diff+ S_less_diff,D_diff+ S_less_n_diff, D_diff+S_less_inv_diff,D_diff+ S_less_inv_n_diff);

            end
                 
             
         %   exp3_diff = diff2_ds_all*ds_pre_inv;
          %  exp3_diff = ds_all(1:Top_boxes-2,:)-exp3_diff;
           % exp3_diff = exp3_diff+D_diff;
            
            deri_diff =  diff2_ds_all*ds_pre_inv;%diff2_ds_all*ds_pre_inv;%diff2_ds_all/ds_pre_inv;% diff_ds_all/ds_pre_inv; %diff2_ds_all*ds_pre_inv;
       %     min_sless = min(S_less_diff(:));
            
         %       D_diff = ds_pre(i,1)+abs(D_diff - abs(min(S_less_diff(:))))*abs());
                 
            
                 
               
               diff_s_less = diff(S_less);
               sol_1 = sum(diff_s_less(:));
               
               S1 = S_less; S1_mean = mean(S1(:),'omitnan');
               S1(S1>S1_mean) = NaN;
               S2 = S1; S2_mean = mean(S2(:),'omitnan');
               S2(S2>S2_mean) = NaN;
               S3 = S2; S3_mean = mean(S3(:),'omitnan');
               S3(S3>S3_mean) = NaN;
               S1(isnan(S1)) = 0;
                S2(isnan(S2)) = 0;
                S3(isnan(S3)) = 0;
               S1_diff = diff(S1);
               S2_diff = diff(S2);
               S3_diff = diff(S3);
               S5 = S3(1:Top_boxes-1,:).*S1_diff;
               
               
               S1(isnan(S1)) = 0;
               S7 = S3(1:Top_boxes-2,:).*diff2_ds_all_less;
               S8 = S7.*ds_pre(i,1); %S_less(1:Top_boxes-1,:);
               S6 = S5.*ds_pre(i,1); %S_less(1:Top_boxes-1,:);
               sol_2 = sum(S1(:));
               sol_3= sum(S2(:));
               sol_4 = sum(S3(:));
               sol_5 = sum(S5(:)); %s_delta_mat(:);
               sol_6 = sum(S8(:)); 
               
               
               Var_S5 = var(S3,1);
               num_var_s5 = nnz(Var_S5);
               sum_var_s5 = sum(Var_S5);
               mum_var_s5 = num_var_s5*sum_var_s5;
               
              
               
              heat3 = diff2_ds_all_less*ds_pre_inv;
              %check_heat = sum(S8(2,:,:));
              check_heat = 0;
              %D_diff = ds_pre(i,1)-; %-s_delta_all;
               for jj = 1:Top_boxes
                   S8_col = S8(:,jj);
                   check_heat_mean = mean(S8_col);

                    S8_col(S8_col<check_heat_mean) = 0;

                    hm = nnz(S8_col);
                    if hm >= 2 
                        check_heat = check_heat+ sum(S8_col);
                    end
               end
               
             for jj = 1:Top_boxes
                  
                   S3_nnz = nnz(S3(:,jj));
                   if S3_nnz < 2
                       sum_diff2_ds_all(jj) = 0;
                       
                   end
                   
                   
               end
               nnz_black_check = nnz(sum_diff2_ds_all);
               
               top_candidates = sum_diff2_ds_all;
               
               check_heat = 0;
%              D_diff = ds_pre(i,1)-; %-s_delta_all;
               for jj = 1:Top_boxes
                   S8_col = S8(:,jj);
                    
                    hm = find(S8_col~=0, 1, 'first');
                    if hm < 2 
                        check_heat = check_heat+ 1;
                    end
                   
               end
               if show_output == 45
                 
                [row,col,value] = find(diff_s_less~=0);
                
                q_imgg = imread(char(qimg_path));
              

                db_imgg = imread(char(db_img));
              
                
                qqq_img = q_imgg;
                dbb_img = db_imgg;
                
                box_var_db = [];
                box_var_q = [];
                
                        for jjj=1:length(row) %Top_boxes

                            qq_img = draw_boxx(q_imgg,imgg_mat_box_q(col(jjj),1:4));%   q_RGB = insertShape(I,'Rectangle',imgg_mat_box_q(row(jjj),1:4),'LineWidth',3);
                            dd_img = draw_boxx(db_imgg,imgg_mat_box_db(row(jjj),1:4));%   q_RGB = insertShape(I,'Rectangle',imgg_mat_box_q(row(jjj),1:4),'LineWidth',3);

                            qqq_img = draw_boxx(qqq_img,imgg_mat_box_q(col(jjj),1:4));%   q_RGB = insertShape(I,'Rectangle',imgg_mat_box_q(row(jjj),1:4),'LineWidth',3);
                            dbb_img = draw_boxx(dbb_img,imgg_mat_box_db(row(jjj),1:4));%   q_RGB = insertShape(I,'Rectangle',imgg_mat_box_q(row(jjj),1:4),'LineWidth',3);


                            bb_q = imgg_mat_box_q(col(jjj),1:4);

                            box_var_q = [box_var_q ; imgg_mat_box_q(col(jjj),1:4) (bb_q(3)+bb_q(1))/2 (bb_q(4)+bb_q(2))/2 ];

                            bb_db = imgg_mat_box_db(row(jjj),1:4);

                            box_var_db = [box_var_db ; imgg_mat_box_db(row(jjj),1:4) (bb_db(3)+bb_db(1))/2 (bb_db(4)+bb_db(2))/2 ];
                          %  subplot(2,2,1); imshow(qq_img); %q_img
                          %  subplot(2,2,2); imshow(dd_img); %



                        end
                std_box_var_q = std(im2double(box_var_q),0,1);
                subplot(2,2,1); bar(std_box_var_q); %q_img
                std_box_var_db = std(im2double(box_var_db),0,1);
                subplot(2,2,2); bar(std_box_var_db); %q_img
                
                  subplot(2,2,3); imshow(qqq_img); %q_img
                    subplot(2,2,4); imshow(dbb_img); %
                end  
               
            if num_var_s5 < 3 
                D_diff = D_diff-mum_var_s5;
            end
              
            if inegatif == 100  && num_var_s5 < 5 
              D_diff = norm(D_diff-sum(S8(:))); 
            end
            
            

             if show_output == 4
                 fprintf(' %f -> %f %f %f %f %f %f %f \n',ds_pre(i,1), D_diff, num_var_s5,sum_var_s5, mum_var_s5,sol_4,sol_5,sol_6);
                 y = [ds_pre(i,1) D_diff sum(S8(:)) inegatif sum(S5(:)) mum_var_s5 num_var_s5 nnz_black_check];
                 q_imgg = imread(char(qimg_path));
                 subplot(2,3,1); imshow(q_imgg); %q_img
                 db_imgg = imread(char(db_img));
                subplot(2,3,2); imshow(db_imgg); %
                
               
                
                subplot(2,3,3); h = heatmap(S8);
                subplot(2,3,4); h = heatmap(y); % with plus is wokring
                subplot(2,3,5); h = heatmap(S3);
              
                
             end
         
             
             
           ds_new_top(i,1) = abs(D_diff);
           

         ds_all = [];
        end
        
        %  SLEN_top(i,1) = i; SLEN_top(i,2) = aa;
          
        [C c_i] = sortrows(ds_new_top);
        idss = ids;
        for i=1:total_top
            idss(i,1) = ids(c_i(i,1));
        end
         if show_output == 3

                subplot(2,6,1); imshow(imread(char(qimg_path))); %q_img
                db_imgo1 = strcat(dataset_path,'/images/', db.dbImageFns{ids(1,1),1});  
                db_imgo2 = strcat(dataset_path,'/images/', db.dbImageFns{ids(2,1),1});  
                db_imgo3 = strcat(dataset_path,'/images/', db.dbImageFns{ids(3,1),1});  
                db_imgo4 = strcat(dataset_path,'/images/', db.dbImageFns{ids(4,1),1});  
                db_imgo5 = strcat(dataset_path,'/images/', db.dbImageFns{ids(5,1),1});  
                db_img1 = strcat(dataset_path,'/images/', db.dbImageFns{idss(1,1),1});  
                db_img2 = strcat(dataset_path,'/images/', db.dbImageFns{idss(2,1),1});  
                db_img3 = strcat(dataset_path,'/images/', db.dbImageFns{idss(3,1),1});
                db_img4 = strcat(dataset_path,'/images/', db.dbImageFns{idss(4,1),1});
                db_img5 = strcat(dataset_path,'/images/', db.dbImageFns{idss(5,1),1});
                
                subplot(2,6,2); imshow(imread(char(db_imgo1))); %
                subplot(2,6,3); imshow(imread(char(db_imgo2))); %
                subplot(2,6,4); imshow(imread(char(db_imgo3))); %
                subplot(2,6,5); imshow(imread(char(db_imgo4))); %
                subplot(2,6,6); imshow(imread(char(db_imgo5))); %
                
                subplot(2,6,8); imshow(imread(char(db_img1))); %
                subplot(2,6,9); imshow(imread(char(db_img2))); %
                subplot(2,6,10); imshow(imread(char(db_img3))); %
                subplot(2,6,11); imshow(imread(char(db_img4))); %
                subplot(2,6,12); imshow(imread(char(db_img5))); %
                fprintf( '==>> %f %f %f %f %f \n',c_i(1,1), c_i(2,1),c_i(3,1), c_i(4,1) ,c_i(5,1));

         end
         
          if show_output == 33

                subplot(2,6,1); imshow(imread(char(qimg_path))); %q_img
                db_imgo1 = strcat(dataset_path,'/images/', db.dbImageFns{idss(1,1),1});  
                db_imgo2 = strcat(dataset_path,'/images/', db.dbImageFns{idss(2,1),1});  
                db_imgo3 = strcat(dataset_path,'/images/', db.dbImageFns{idss(3,1),1});
                db_imgo4 = strcat(dataset_path,'/images/', db.dbImageFns{idss(4,1),1});
                db_imgo5 = strcat(dataset_path,'/images/', db.dbImageFns{idss(5,1),1});
                db_img1 = strcat(dataset_path,'/images/', db.dbImageFns{idss(6,1),1});  
                db_img2 = strcat(dataset_path,'/images/', db.dbImageFns{idss(7,1),1});  
                db_img3 = strcat(dataset_path,'/images/', db.dbImageFns{idss(8,1),1});  
                db_img4 = strcat(dataset_path,'/images/', db.dbImageFns{idss(9,1),1});  
                db_img5 = strcat(dataset_path,'/images/', db.dbImageFns{idss(10,1),1});  
                
                subplot(2,6,2); imshow(imread(char(db_imgo1))); %
                subplot(2,6,3); imshow(imread(char(db_imgo2))); %
                subplot(2,6,4); imshow(imread(char(db_imgo3))); %
                subplot(2,6,5); imshow(imread(char(db_imgo4))); %
                subplot(2,6,6); imshow(imread(char(db_imgo5))); %
                
                subplot(2,6,8); imshow(imread(char(db_img1))); %
                subplot(2,6,9); imshow(imread(char(db_img2))); %
                subplot(2,6,10); imshow(imread(char(db_img3))); %
                subplot(2,6,11); imshow(imread(char(db_img4))); %
                subplot(2,6,12); imshow(imread(char(db_img5))); %
                fprintf( '==>> %f %f %f %f %f %f %f %f %f %f \n',c_i(1,1), c_i(2,1),c_i(3,1), c_i(4,1) ,c_i(5,1), c_i(6,1), c_i(7,1),c_i(8,1), c_i(9,1) ,c_i(10,1));

         end
        
        
        iTestSample
        %% LEO END
            
            
        numReturned= length(ids);
        assert(numReturned<=nTop); % if your searcher returns fewer, it's your fault
        
        thisRecall= cumsum( isPos(iTest, idss) ) > 0; % yahan se get karta hai %db.cp (close position)
        recalls(iTestSample, :)= thisRecall( min(ns, numReturned) );
        
        thisRecall1= cumsum( isPos(iTest, ids) ) > 0; % yahan se get karta hai %db.cp (close position)
        recalls_ori(iTestSample, :)= thisRecall1( min(ns, numReturned) );
        printRecalls(iTestSample)= thisRecall(printN);
        
        thisRecall_idx = find(thisRecall~=0, 1, 'first');
        thisRecall1_idx = find(thisRecall1~=0, 1, 'first');
        
        if thisRecall_idx-thisRecall1_idx > 1
           fprintf('iTestSample: %i \n',iTestSample);
        end
        
        if thisRecall(1) == 0
          fprintf('iTestSample: %i \n',iTestSample);
  %           plot(ns, recalls(1:iTestSample,:), 'ro-',ns, recalls_ori(1:iTestSample,:), 'go-'); grid on; xlabel('N'); ylabel('Recall@N'); title('Tokyo247 HYBRID Edge Image', 'Interpreter', 'none');

        end
       
    end
    t= toc(evalProg);
    
    res= mean(printRecalls);
    relja_display('\n\trec@%d= %.4f, time= %.4f s, avgTime= %.4f ms\n', printN, res, t, t*1000/length(toTest));
    
    relja_display('%03d %.4f\n', [ns(:), mean(recalls,1)']');
    
    rng(rngState);
end

   
           % S_less_n = S_less - ds_all_less_mean;
           % S_less_diff = diff(S_less);
            
%             ipositif_inv=sum(s_inv(:)==1);
%             inegatif_inv=sum(s_inv(:)==-1);
%             S_great_inv = s_inv; S_great_inv(S_great_inv<0) = 0; S_great_inv = S_great_inv.*ds_all_less_inv; S_great_inv_n = S_great_inv - ds_all_less_inv_mean; 
%             S_less_inv = s_inv; S_less_inv(S_less_inv>0) = 0; S_less_inv = abs(S_less_inv).*ds_all_less_inv; S_less_inv_n = S_less_inv - ds_all_less_inv_mean;
% 
%             %  [S_less_min_inv, S_less_I_inv] = sort(S_less_inv(:));
%             
% 
%            S_great_mean = sum(S_great(:)/ipositif); S_great_n_mean = sum(S_great_n(:)/ipositif);
%            S_great_inv_mean = sum(S_great_inv(:)/ipositif_inv); S_great_inv_n_mean = sum(S_great_inv_n(:)/ipositif_inv);
%            
           
           
           
%            S_less_mean = sum(sum(S_less/inegatif)); S_less_n_mean = sum(S_less_n(:)/inegatif);
%            S_less_inv_mean = sum(S_less_inv(:)/inegatif_inv); S_less_inv_n_mean = sum(S_less_inv_n(:)/inegatif_inv);
%           
%              
%            S_less_diff = diff(S_less); 
%            S_less_n_diff = sum(sum(S_less_n.*diff_ds_all));
%            S_less_inv_diff = sum(sum(S_less_inv.*diff_ds_all)); S_less_inv_n_diff = sum(sum(S_less_inv_n.*diff_ds_all));
%           
           

           % subplot(2,2,3); h = heatmap(S_less.*diff_ds_all);
           % subplot(2,2,4); h = heatmap(S_less_n.*diff_ds_all);
            
           % subplot(2,2,3); h = heatmap(S_less_inv);
           % subplot(2,2,4); h = heatmap(S_less_inv_n);
            
           
          
          
            
            
%             D = sum(sum(S_less(1:Top_boxes)));
           %  boxes_per_less = (Top_boxes*Top_boxes)/inegatif;

            
         %   exp3_diff = diff2_ds_all*ds_pre_inv;
          %  exp3_diff = ds_all(1:Top_boxes-2,:)-exp3_diff;
           % exp3_diff = exp3_diff+D_diff;
            
%             deri_diff =  diff2_ds_all*ds_pre_inv;%diff2_ds_all*ds_pre_inv;%diff2_ds_all/ds_pre_inv;% diff_ds_all/ds_pre_inv; %diff2_ds_all*ds_pre_inv;
       %     min_sless = min(S_less_diff(:));
            
         %       D_diff = ds_pre(i,1)+abs(D_diff - abs(min(S_less_diff(:))))*abs());
                 
            
%                diff_s_less = diff(S_less);
%                sol_1 = sum(diff_s_less(:));

function plot_mat(A)
lowestValue = min(A(A(:)>0));
highestValue = max(A(:));
imagesc(A);
cmap = jet(256);
colormap(cmap);
caxis(gca,[lowestValue-2/256, highestValue]);
% Make less than lowest value black:
cmap(1,:)=[0,0,0];
colormap(cmap)
caxis([-0.2 0.2]);
colorbar
end

function [mat_boxes,im, edge_image, wd, hh] = img_Bbox(db_img,model)
im= vl_imreadjpeg({char(db_img)},'numThreads', 12); 
I = uint8(im{1,1});
[bbox, E] =edgeBoxes(I,model);
[wd, hh] = size(im{1,1});
edge_image = uint8(E * 255);
bboxes=[];
gt=[111	98	25	101];

b_size = size(bbox,1); 
for ii=1:b_size
     bb=bbox(ii,:);
     square = bb(3)*bb(4);
     if square <2*gt(3)*gt(4)
        bboxes=[bbox;bb];
     end
end

mat_boxes = uint8(bboxes); 
end

function img = draw_boxx(I,bb)

bb=[bb(1) bb(2) bb(3)+bb(1) bb(4)+bb(2)];

img = insertShape(I,'Rectangle',bb,'LineWidth',3);

end
