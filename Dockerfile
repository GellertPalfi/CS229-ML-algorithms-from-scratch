FROM python:3.10-slim
WORKDIR /usr/src/app
RUN apt-get update && \
    apt-get install -y git
COPY . .
RUN pip install --upgrade -q pip && pip install -r requirements.txt -q

