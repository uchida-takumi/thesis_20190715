# set docker image
FROM python:3

# set working directory
WORKDIR /docker_work

# add all file to docker
ADD . .

# RUN pip install -R requirements.txt
RUN pip install -r requirements.txt

# exec command
CMD [ "python", "-V" ]
