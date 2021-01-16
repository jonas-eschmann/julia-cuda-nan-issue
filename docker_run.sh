#!/usr/bin/env bash

docker build -t julia-cuda-crash .
docker run -it --rm --gpus all -v $(pwd):/mounted -w /mounted julia-cuda-crash julia --project=/build crash.jl
