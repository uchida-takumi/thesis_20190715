# set docker image
#FROM python:3
FROM tensorflow/tensorflow:2.0.0b0-gpu-py3-jupyter

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

# RUN install requirements
RUN pip install -r requirements.txt
RUN pip install surprise
RUN pip install src/pyFM/
RUN pip install tensorflow-gpu==2.0.0-beta1

# exec command
CMD [ "python", "-V" ]
