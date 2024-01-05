FROM python:3.10-slim
WORKDIR /usr/src/app

COPY . .
RUN pip install --upgrade -q pip && pip install -r requirements.txt -q

