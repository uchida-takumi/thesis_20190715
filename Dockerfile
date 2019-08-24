# set docker image
#FROM python:3
#FROM tensorflow/tensorflow:2.0.0b0-gpu-py3-jupyter
FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04

# set working directory
WORKDIR /docker_work

# add all file to docker
ADD . .

# download data
#RUN mkdir data
#RUN cd data
## MovieLens small data
#RUN curl -OL http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
#RUN unzip ml-latest-small.zip
## MovieLens 20m data (recommended for research)
#RUN curl -OL http://files.grouplens.org/datasets/movielens/ml-20m.zip
#RUN unzip ml-20m.zip
#RUN cd ..

# set up
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y upgrade python3
RUN apt-get -y install python3-pip
RUN apt-get install -y --no-install-recommends build-essential 
#RUN apt-get clean 
#RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip #can use pip, but can not use pip3.
RUN pip --no-cache-dir install -U setuptools    

# RUN install requirements
RUN pip install -r requirements.txt
RUN pip install surprise
RUN pip install src/pyFM/
RUN pip install tensorflow-gpu==2.0.0-beta1

# set environment 
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# exec command
CMD [ "python3", "-V" ]
