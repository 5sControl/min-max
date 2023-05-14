FROM python:3.9
RUN apt-get update
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /var/www/5scontrol
COPY . .
RUN mkdir -p /usr/src/app
ENTRYPOINT ["python", "-u", "main.py"]