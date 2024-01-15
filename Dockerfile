FROM python:3.10-slim
WORKDIR /usr/src/app
RUN apt-get -qq update && \
    apt-get -qq install -y git

RUN pip install --upgrade -q pip && pip install -r requirements.txt -q
COPY . .
