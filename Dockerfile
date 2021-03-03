FROM python:3.8

ENV LANG C.UTF-8
ENV PYTHONUNBUFFERED 1
ENV SOURCE_DIR /opt/src
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx

RUN mkdir $SOURCE_DIR
WORKDIR $SOURCE_DIR

COPY requirements.txt $SOURCE_DIR

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 freeze

ADD . $SOURCE_DIR