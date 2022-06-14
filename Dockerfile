FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY api/ api/
COPY src/ src/
WORKDIR  ./api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
