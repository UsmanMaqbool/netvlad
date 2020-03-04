!find . -name '*.mexa64' -exec rm {} \;
%export LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/4.8/libgomp.so
mex private/edgesDetectMex.cpp -outdir private '-DUSEOMP' CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
mex private/edgesNmsMex.cpp    -outdir private '-DUSEOMP' CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
mex private/spDetectMex.cpp    -outdir private '-DUSEOMP' CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
mex private/edgeBoxesMex.cpp   -outdir private

addpath(pwd);
