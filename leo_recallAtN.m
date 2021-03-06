 function [res, recalls]= leo_recallAtN(searcher, nQueries, isPos, ns, printN, nSample,db)
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
    
    %% LEO START
    
    netID= 'vd16_tokyoTM_conv5_3_vlad_preL2_intra_white';
    % netID= 'caffe_tokyoTM_conv5_vlad_preL2_intra_white';


    paths= localPaths();

    load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );

    %%
    net= relja_simplenn_tidy(net); % potentially upgrate the network to the latest version of NetVLAD / MatConvNet

    
    
    
    
    addpath(genpath('/mnt/02/docker_ws/docker_ws/netvlad/slen-0.2-box'));
    
    %% EDGE BOX
    %load pre-trained edge detection model and set opts (see edgesDemo.m)

    model=load('models/forest/modelBsds'); model=model.model;
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

    
    
    
    top_100 = [];
    
    dataset_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV';
    save_path = '/home/leo/docker_ws/datasets/vt-2';
    
    %dataset_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV';
    %save_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV/vt-2';
    show_output = 0; startfrom = 1;

    for iTestSample= 1:length(toTest)
        
        %Display
        relja_progress(iTestSample, ...
                       length(toTest), ...
                       sprintf('%.4f', mean(printRecalls(1:(iTestSample-1)))), evalProg);
        
     
        iTest= toTest(iTestSample);
        
        [ids ds_pre]= searcher(iTest, nTop); % Main function to find top 100 candidaes
        
       
        %% Leo START
                
        qimg_path = strcat(dataset_path,'/query/', db.qImageFns{iTestSample, 1});  
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
        else
            im= vl_imreadjpeg({char(qimg_path)},'numThreads', 12); 

            I = uint8(im{1,1});
            [bbox, E] =edgeBoxes(I,model);
            [wd, hh] = size(im{1,1});
            mat_boxes = leo_slen_increase_boxes(bbox,wd,hh);

            im= im{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
            query_full_feat= leo_computeRepresentation(net, im, mat_boxes); % add `'useGPU', false` if you want to use the CPU

            save(q_feat,'query_full_feat');
        end
        
        total_top = 100; %100;
 
        q_dbfeat = strrep(q_feat,'.mat','_db_feats.mat');
        if exist(q_dbfeat, 'file')
            load(q_dbfeat);
        else
            % Top 100 sample
            for jj = 1:total_top

                    db_img = strcat(dataset_path,'/images/', db.dbImageFns{ids(jj,1),1});  
                    im= vl_imreadjpeg({char(db_img)},'numThreads', 12); 
                    I = uint8(im{1,1});
                    [bbox, E] =edgeBoxes(I,model);
                    [wd, hh] = size(im{1,1});
                    mat_boxes = leo_slen_increase_boxes(bbox,wd,hh);

                    im= im{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
                    feats= leo_computeRepresentation(net, im, mat_boxes); % add `'useGPU', false` if you want to use the CPU
                    feats_file(jj) = struct ('featsdb', feats); 
                    clear feats;
                    fprintf( '==>> %i ~ %i/%i ',jj,iTestSample,total_top );

            end
            save(q_dbfeat,'feats_file');
            
        end
        SLEN_top = zeros(total_top,2); 
        Top_boxes = 10;
        k = Top_boxes;
        ds_all = [];
       % figure;

        for i=startfrom:total_top 
            feats2 = feats_file(i).featsdb;
            for j = 1:Top_boxes
                q1 = single(feats2(:,j));  %take column of each box
                [ids1, ds1]= yael_nn(query_full_feat, q1, k);
                ds_all = [ds_all ds1];
            end
            % original dis: 1.25 ds_pre
            db_img = strcat(dataset_path,'/images/', db.dbImageFns{ids(i,1),1});  
            ds_pre_inv = 1/ds_pre(i,1);
            ds_all_diff = 1./(ds_all);
            ds_all_deri = diff(ds_all);
            
           
           %ds_all_sub = ds_all_diff(1:10,1:10);
            ds_all_sub = ds_all(1:Top_boxes,1:Top_boxes);
            
            ds_all_less = ds_all-ds_pre(i,1);  
            ds_all_less = ds_all_less(1:Top_boxes,1:Top_boxes);
            ds_all_diff = ds_all_diff(1:Top_boxes,1:Top_boxes);
            ds_all_less_mean = mean(ds_all_less(:));
            s=sign(ds_all_less);
            ipositif=sum(s(:)==1);
            inegatif=sum(s(:)==-1);
            S_great = s; S_great(S_great<0) = 0; S_great = S_great.*ds_all_sub; %S_great = sum(sum(S_great));
            S_less = s; S_less(S_less>0) = 0; S_less = abs(S_less).*ds_all_sub; %S_less = sum(sum(S_less));
            S_less_diff = diff(S_less);
            [S_less_min, S_less_I] = sort(S_less(:));
            s_less_min_diff = ds_pre(i,1)-S_less_min;

           % ds_all_less_mean = sum(S_less(:)/inegatif);
            ds_pre(i,1);
            s_delta_all = 0;
    
            
            s_delta_mat = 0;
            s_dis = 0;
            for jj = 1:Top_boxes
                S_less_col = S_less(:,jj);
                s_near_mat = [];
                for jjj = 1:Top_boxes-1
                                           
                        s_dis = ds_pre(i,1) - S_less_col(jjj);
                        s_less_difference = S_less_col(jjj+1)-S_less_col(jjj);
                        if s_less_difference > 0 && s_less_difference <= 0.01 && s_dis > .24
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

                subplot(2,2,3); h = heatmap(S_less);
                subplot(2,2,4); h = heatmap(S_less_diff);
%                fprintf( '==>> Distance %f ~ Greator Values %f %f \n Less Values %f %f ~ Min %f \n',ds_pre(i,1), s_delta_all,ipositif, S_great, inegatif, S_less);
              %  fprintf('%f %f %f %f %f %f %f %f %f %f\n',(Top_boxes*Top_boxes)/inegatif, D_diff, D_diff+ds_all_less_mean, D_diff-(ds_all_less_mean/D), D_diff-s_delta_mat, D_diff+s_delta_mat,ds_all_less_mean+s_delta_mat, s_delta_mat/ds_all_less_mean, D/(D_diff-ds_all_less_mean), D);
                fprintf('\n For %f, %f %f %f %f', Top_boxes, inegatif, s_delta_mat, ds_all_less_mean, mean(S_less_diff(:)));

            end
            
            
            
            if inegatif == 100 && mean(S_less_diff(:)) < 1.79
                
                D_diff = D_diff+(ds_all_less_mean); %Top_boxes0/(D*ds_pre_inv);
            else
                D_diff = D_diff-abs(s_delta_mat)+ds_all_less_mean; % /boxes_per_less); %100/(D*ds_pre_inv);
            end
          % D_diff = 1/D;
          
           ds_new_top(i,1) = abs(D_diff);
         %    fprintf('~ %f -> %f for %f \n',ds_pre(i,1), D_diff, i);

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
        
        
        iTestSample
        %% LEO END
            
            
        numReturned= length(ids);
        assert(numReturned<=nTop); % if your searcher returns fewer, it's your fault
        
        thisRecall= cumsum( isPos(iTest, idss) ) > 0; % yahan se get karta hai %db.cp (close position)
        recalls(iTestSample, :)= thisRecall( min(ns, numReturned) );
        
        thisRecall1= cumsum( isPos(iTest, ids) ) > 0; % yahan se get karta hai %db.cp (close position)
        recalls_ori(iTestSample, :)= thisRecall1( min(ns, numReturned) );
        printRecalls(iTestSample)= thisRecall(printN);
        if thisRecall(1) == 0
  %          fprintf('iTestSample: %i \n',iTestSample);
  %           plot(ns, recalls(1:iTestSample,:), 'ro-',ns, recalls_ori(1:iTestSample,:), 'go-'); grid on; xlabel('N'); ylabel('Recall@N'); title('Tokyo247 HYBRID Edge Image', 'Interpreter', 'none');

        end
       
    end
    t= toc(evalProg);
    
    res= mean(printRecalls);
    relja_display('\n\trec@%d= %.4f, time= %.4f s, avgTime= %.4f ms\n', printN, res, t, t*1000/length(toTest));
    
    relja_display('%03d %.4f\n', [ns(:), mean(recalls,1)']');
    
    rng(rngState);
end

