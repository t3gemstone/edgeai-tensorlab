FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        bash \
        build-essential \
        curl \
        software-properties-common \
        git \
        locales \
        sudo \
        chrpath \
        libbz2-dev \
        libffi-dev \
        liblzma-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libxml2-dev \
        libxmlsec1-dev \
        llvm make \
        tk-dev \
        xz-utils \
        wget \
        curl \
        libffi-dev \
        libjpeg-dev \
        zlib1g-dev \
        graphviz \
        graphviz-dev \
        protobuf-compiler \
        python3 \
        python3-venv \
        python3-pip

# Allow minimum password length of image in Distrobox to be 1 character
RUN sed -i 's/pam_unix\.so obscure/pam_unix.so minlen=1 obscure/' /etc/pam.d/common-password
RUN echo gemstone > /etc/hostname

# Taskfile Installation
RUN curl --location https://taskfile.dev/install.sh | sudo sh -s -- -d -b /usr/local/bin && \
    task --completion bash > /etc/bash_completion.d/task

CMD ["bash"]