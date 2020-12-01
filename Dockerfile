# This is a simple Dockerfile to use while developing
# It's not suitable for production
#
FROM python:3.8

LABEL maintainer="adam2392@gmail.com"

# install python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install pipenv

# create code directory to store app
RUN mkdir /researcher
WORKDIR /home/researcher

# copy over the API code
COPY mtsmorf tmp/mtsmorf/

# run the installation of REST api app
COPY Pipfile* tmp/
RUN cd tmp/ && pipenv install --system --deploy --ignore-pipfile
COPY setup.* tmp/
COPY README.md tmp/
RUN cd tmp/ && pip install -e .
