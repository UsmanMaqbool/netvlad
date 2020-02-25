%% clc;
clear all;
setup;

%%
paths= localPaths();

%%

db = dbTokyo247();

images_q = db.qImageFns;
images_paths = paths.dsetRootTokyo247;
save_paths = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV';
copy_test_files(images_paths,save_paths,images_q);


%images  db.qImageFns;

function copy_test_files(p1,pdest,filename)
for i = 1:size(filename)
        source = fullfile(p1,filename(i));  
        
    if exist(char(source), 'file')
        
        destination = fullfile(pdest,filename(i));
         [folder, baseFileName, extension] = fileparts(char(destination));
        if exist(folder, 'dir')==0
            mkdir(char(folder))
        end
        copyfile(char(source),char(destination));
        fprintf( '  ==>> %i/%i \n',i,length(filename));
    else
        break;
        filename(i)
    end
end
end