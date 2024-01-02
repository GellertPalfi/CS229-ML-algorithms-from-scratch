FROM python:3.10-slim
WORKDIR /usr/src/app

RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
COPY . .
RUN pip install --upgrade -q pip && pip install -r requirements.txt -q

