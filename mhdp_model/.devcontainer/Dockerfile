
FROM continuumio/miniconda3

# Store the provided MD5SUM of the source Dockerfile
ARG MD5SUM
RUN echo ${MD5SUM} > /DOCKERFILE_MD5 ; cat /DOCKERFILE_MD5

# USER_NAME shouldn't be changed, please leave it as vscode
ENV USER_NAME vscode
ENV PROJECT mhd_model

# Preset timezone info
ENV TZ=Europe/Dublin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# Select a suitable UTF-8 locale
# RUN locale-gen en_US.UTF-8
# ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

SHELL ["/bin/bash", "-c"]
