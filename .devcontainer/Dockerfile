# Base image containing julia
FROM debian:bookworm

# Install git
RUN apt-get update && apt-get install -y git

# Install `pipx` and ensure its directory is in the PATH. I think this also installs some `python3`
RUN apt-get -y install pipx
RUN pipx ensurepath

# Install `poetry` with `pipx`
RUN pipx install poetry==2.0.1

# `llvm` is needed for `numba` and `llvmlite`
RUN apt-get update && apt-get install -y build-essential llvm

# install `julia`
# TODO: the installation script does not support specifying the version; it was better if it did so I could specify the version here; last time I checked, this installed 1.10.8
RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://install.julialang.org | sh -s -- --default-channel lts --yes