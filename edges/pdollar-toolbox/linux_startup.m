!find . -name '*.mexa64' -exec rm {} \;

toolboxCompile
addpath(genpath(pwd)); savepath;
