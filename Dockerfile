FROM julia-gpu
COPY *.toml /build/
WORKDIR /build
RUN julia --project=. -e "using Pkg; Pkg.instantiate();"
RUN julia --project=. -e "using Pkg; Pkg.precompile();"
