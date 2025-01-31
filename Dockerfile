FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV FLASK_APP app.py
ENV FLASK_HOST 0.0.0.0

CMD ["python", "app.py"]
