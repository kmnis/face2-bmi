FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Expose port you want your app on
EXPOSE 8501

# Upgrade pip and install requirements
COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

RUN mkdir /app/src /app/models
COPY src/app.py /app/src/app.py
COPY models/ /app/models/

ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "–server.address=0.0.0.0"]
