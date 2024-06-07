FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y tzdata \
    && ln -fs /usr/share/zoneinfo/Europe/Madrid /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

ENV TZ=Europe/Madrid

# Install python3.11.7
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv
RUN apt-get install -y nginx

# Install curl and network tools
RUN apt-get install -y curl net-tools

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.11 get-pip.py

# Install gcc and git
RUN apt-get update && apt-get install -y build-essential gcc g++ clang git make cmake g++-aarch64-linux-gnu dos2unix

# Install docker
RUN apt-get install -y ca-certificates curl gnupg
RUN install -m 0755 -d /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
RUN chmod a+r /etc/apt/keyrings/docker.gpg
RUN echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN apt-get update

RUN apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin


COPY requirements.txt .

# Install the required packages
RUN python3.11 -m pip install --ignore-installed -r requirements.txt

# Configure Nginx
# RUN rm /etc/nginx/sites-enabled/default
# COPY /config/nebula /etc/nginx/sites-available/
# RUN ln -s /etc/nginx/sites-available/nebula /etc/nginx/sites-enabled/

COPY start_services.sh .

RUN dos2unix start_services.sh

RUN chmod +x start_services.sh

ENTRYPOINT ["/bin/bash", "/start_services.sh"]