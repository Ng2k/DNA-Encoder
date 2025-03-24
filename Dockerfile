# Base image with CUDA 12.8.1+ and Python 3.10
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Set environment variables for GPU support
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
ENV CUDA_VISIBLE_DEVICES=0