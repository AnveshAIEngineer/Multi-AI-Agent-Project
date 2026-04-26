# # Use the Jenkins image as the base image
# FROM jenkins/jenkins:lts

# # Switch to root user to install dependencies
# USER root

# # Install prerequisites and Docker
# RUN apt-get update -y && \
#     apt-get install -y ca-certificates curl gnupg docker.io && \
#     apt-get clean

# # Add Jenkins user to the Docker group (create if it doesn't exist)
# RUN groupadd -f docker && \
#     usermod -aG docker jenkins

# # Create the Docker directory and volume for DinD
# RUN mkdir -p /var/lib/docker
# VOLUME /var/lib/docker

# # Switch back to the Jenkins user
# USER jenkins