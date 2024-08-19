 # Use an official PyTorch image as a parent image
FROM pytorch/pytorch:latest

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set the environment variable to avoid the creation of .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# Set the environment variable to ensure the output is shown in the terminal immediately
ENV PYTHONUNBUFFERED 1

# Upgrade system dependencies
RUN apt-get update \
    && apt-get install xz-utils \
    && apt-get -y install curl \
    # Verify git, needed tools installed
    && apt-get -y install git iproute2 procps curl lsb-release \
    && apt install -y sudo

# Set the working directory first to ensure that files are copied to the correct location
WORKDIR /workspace

# Copy the requirements.txt file into the container
COPY requirements.txt /workspace/

# Install any necessary dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /workspace

# Expose port 8080
EXPOSE 8080