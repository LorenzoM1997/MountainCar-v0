FROM ubuntu:18.04

#install some dependencies
RUN apt-get update && \
  apt-get install -y \
	python3 \
  python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install gym==0.10.11
RUN pip3 install imageio==2.4.0
RUN pip3 install PILLOW
RUN pip3 install pyglet==1.3.2
RUN pip3 install tensorflow==2.1.0
RUN pip3 install tf-agents
RUN pip3 install matplotlib
