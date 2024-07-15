FROM continuumio/miniconda3

WORKDIR /app
COPY requirements.txt .

RUN apt-get update
RUN apt-get install nano unzip
RUN apt install curl -y

RUN curl -fsSL https://get.deta.dev/cli.sh | sh
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
EXPOSE 8000
#CMD gunicorn app:app  --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker  --timeout 600
#CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["uvicorn", "run", "--server.port=8000", "app.py"]
#uvicorn APP:app --host 0.0.0.0 --port PORT
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
