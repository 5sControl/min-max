FROM python:3.10
RUN apt-get update
COPY requirements.txt .
RUN pip --default-timeout=100 install -r requirements.txt
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /var/www/5scontrol
COPY . .
RUN mkdir -p /usr/src/app
ENTRYPOINT ["python", "-u", "main.py"]