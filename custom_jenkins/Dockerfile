FROM jenkins/jenkins:lts

USER root

RUN apt-get update && \
    apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    docker.io && \
    apt-get clean

USER jenkins