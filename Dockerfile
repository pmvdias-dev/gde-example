FROM python:3.14-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates/ templates/
COPY static/ static/

# Chat history lives on a mounted volume so it survives container replacement.
ENV DATABASE=/data/chat.db
RUN mkdir /data && useradd -r -m app && chown app /data
USER app

# No AWS credentials in the image: boto3 picks up the EC2 instance role at runtime.
EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "-w", "2", "app:app"]
