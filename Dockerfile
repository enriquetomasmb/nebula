FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y tzdata \
    && ln -fs /usr/share/zoneinfo/Europe/Madrid /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

ENV TZ=Europe/Madrid

# Install python3.11.7
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv

# Install curl and network tools
RUN apt-get install -y curl net-tools iproute2 iputils-ping

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.11 get-pip.py

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip

# Install gcc and git
RUN apt-get update && apt-get install -y build-essential gcc g++ clang git make cmake

WORKDIR /nebula
COPY requirements.txt .
# Install the required packages
RUN python3.11 -m pip install --ignore-installed -r requirements.txt