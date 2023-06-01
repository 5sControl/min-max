FROM python:3.10
RUN apt-get update
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /var/www/5scontrol
COPY . .
RUN mkdir -p /usr/src/app
RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
ENTRYPOINT ["python", "-u", "run.py"]
