%% clc;
clear all;
setup;

%%
paths= localPaths();

%%

%db = dbTokyo247();
db= dbPitts('30k','test');

images = db.dbImageFns;
images_paths = char('/home/leo/docker_ws/datasets/Pittsburgh-all/Pittsburgh/images/'); % paths.dsetRootTokyo247;
save_paths = '/home/leo/docker_ws/datasets/Test_Pitts30k/images/';
%images_q = db.qImageFns;
%images_paths = char('/home/leo/docker_ws/datasets/Pittsburgh-all/Pittsburgh/queries/'); % paths.dsetRootTokyo247;
%save_paths = '/home/leo/docker_ws/datasets/Test_Pitts30k/queries/';
copy_test_files(images_paths,save_paths,images);


%images  db.qImageFns;

function copy_test_files(p1,pdest,filename)
for i = 63048:size(filename)
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