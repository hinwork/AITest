FROM python:3.10 - slim

WORKDIR / app


RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PORT=8080


CMD gunicorn --bind :$PORT --workers 1 --threads 8 app:app
