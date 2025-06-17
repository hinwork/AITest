FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python -c "from transformers import BertTokenizer, BertForSequenceClassification; BertTokenizer.from_pretrained('bert-base-chinese'); BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)"


ENV PORT 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
