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
    
    dataset_path = '/mnt/02/docker_ws/datasets/test-vt/247dataset/247_Tokyo_GSV';
    save_path = '/mnt/02/docker_ws/datasets/test-vt/247dataset/247_Tokyo_GSV/vt-2';
    
    %dataset_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV';
    %save_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV/vt-2';
    for iTestSample= 1:length(toTest)
        
        %Display
        relja_progress(iTestSample, ...
                       length(toTest), ...
                       sprintf('%.4f', mean(printRecalls(1:(iTestSample-1)))), evalProg);
        
     
        iTest= toTest(iTestSample);
        
        ids= searcher(iTest, nTop); % Main function to find top 100 candidaes
          
        
        %% Leo START
       % tt = sort(ids(:,1)); %sort karna hai aisy k maza a jay 
        
            
        qimg_path = strcat(dataset_path,'/query/', db.qImageFns{iTestSample, 1});  
        q_img = strcat(save_path,'/', db.qImageFns{iTestSample, 1});  
        q_feat = strrep(q_img,'.jpg','.mat');
        if exist(q_feat, 'file')
            load(q_feat);
        else
            im= vl_imreadjpeg({char(qimg_path)}); 

            I = uint8(im{1,1});
            [bbox, E] =edgeBoxes(I,model);
            [wd, hh] = size(im{1,1});
            mat_boxes = leo_slen_increase_boxes(bbox,wd,hh);

            im= im{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
            query_full_feat= leo_computeRepresentation(net, im, mat_boxes); % add `'useGPU', false` if you want to use the CPU

            save(q_feat,'query_full_feat');
        end
        
        total_top = 100; %100;
 
        
        
        min_old = 0; max_old = 0; 
       
        q_dbfeat = strrep(q_feat,'.mat','_db_feats.mat');
        if exist(q_dbfeat, 'file')
            load(q_dbfeat);
        else
            % Top 100 sample
            for jj = 1:total_top

                    db_img = strcat(dataset_path,'/images/', db.dbImageFns{ids(jj,1),1});  
                    im= vl_imreadjpeg({char(db_img)}); 
                    I = uint8(im{1,1});
                    [bbox, E] =edgeBoxes(I,model);
                    [wd, hh] = size(im{1,1});
                    mat_boxes = leo_slen_increase_boxes(bbox,wd,hh);

                    im= im{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
                    feats= leo_computeRepresentation(net, im, mat_boxes); % add `'useGPU', false` if you want to use the CPU
                    feats_file(jj) = struct ('featsdb', feats); 
                    clear feats;
                    fprintf( '==>> %i ~ %i \n ',jj,total_top );

            end
            save(q_dbfeat,'feats_file');
            
        end
        SLEN_top = zeros(total_top,2); 
        Top_boxes = 50;
        k = Top_boxes;
        
        for i=1:total_top
            feats2 = feats_file(i).featsdb;
            for j = 1:Top_boxes
                q1 = single(feats2(:,j));  %take column of each box
                [ids1, ds1]= yael_nn(query_full_feat, q1, k);
             
            end
            % original dis: 1.25
            % Spatial Dis:
            y=sort(ds1(:),'ascend');
            aa = sum(y(1:Top_boxes))/Top_boxes;
            SLEN_top(i,1) = i; SLEN_top(i,2) = aa;
          
        end

        C = sortrows(SLEN_top,2);
        idss = ids;
        for i=1:total_top
            idss(i,1) = ids(C(i,1));
        end
        %% LEO END
            
            
        numReturned= length(ids);
        assert(numReturned<=nTop); % if your searcher returns fewer, it's your fault
        
        thisRecall= cumsum( isPos(iTest, idss) ) > 0; % yahan se get karta hai %db.cp (close position)
        recalls(iTestSample, :)= thisRecall( min(ns, numReturned) );
        printRecalls(iTestSample)= thisRecall(printN);
    end
    t= toc(evalProg);
    
    res= mean(printRecalls);
    relja_display('\n\trec@%d= %.4f, time= %.4f s, avgTime= %.4f ms\n', printN, res, t, t*1000/length(toTest));
    
    relja_display('%03d %.4f\n', [ns(:), mean(recalls,1)']');
    
    rng(rngState);
end

