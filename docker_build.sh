#!/usr/bin/env bash
# Create->Start->Commit instead of Dockerfile based because CUDA.jl needs the GPU to be available when probing the driver version to download the correct version of the CUDNN artifacts etc.
container=$(docker create --gpus all julia julia -e "using Pkg;Pkg.add(\"CUDA\"); using CUDA; CUDA.version()") 
docker start -ia $container
docker commit $container julia-gpu
docker rm $container
