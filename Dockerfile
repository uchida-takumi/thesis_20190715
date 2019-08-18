# set docker image
#FROM python:3
FROM tensorflow/tensorflow:2.0.0b0-gpu-py3-jupyter

# set working directory
WORKDIR /docker_work

# add all file to docker
ADD . .

# RUN install requirements
RUN pip install -r requirements.txt
RUN pip install surprise
RUN pip install src/pyFM/
RUN pip install tensorflow-gpu==2.0.0-beta1


# exec command
CMD [ "python", "-V" ]
