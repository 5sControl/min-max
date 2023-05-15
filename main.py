from fastapi import Request
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import requests
import re
import os
import json
from sys import stdout
from detect_min_max import run_min_max
import asyncio
import random
from typing import Union
from fastapi_utils.tasks import repeat_every


app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



algorithms = {"idle_control": {"file": "detect_idle.py", "weights": "yolov7-tiny.pt",  "box_model": "min_max_v0.2.6.pt", "human_model" : "yolov7x.pt"},
              "min_max_control": {"file": "detect_idle.py", "box_model": "min_max_v0.2.6.pt", "human_model" : "yolov7.pt", "algorithm": run_min_max}
}

running_algorithms = []
pids = {}
cameras = {}
@app.on_event("startup")
@repeat_every(seconds=20, wait_first=True)
def interval():
    print(running_algorithms, 'running_algorithms')



class Data(BaseModel):
    algorithm: str
    camera_url: str
    server_url: str
    extra: list = []


class StopData(BaseModel):
    pid: int
# python -u detect_min_max.py --weights min_max_v0.0.4.pt --nosave --source http://192.168.1.110:3456/onvif-http/snapshot?Profile_1 --device cpu --classes 0 --server_url 192.168.1.110 --ip_address 192.168.1.110 --algorithm min_max_control --username admin --password just4Taqtile


@app.post("/run", status_code=201)
async def run(data: Data):
    global pids
    global algorithms
    global cameras
    global running_algorithms
    camera_url = data.camera_url
    try:
        ip_address = re.findall(r'(?:\d{1,3}\.)+(?:\d{1,3})', camera_url)[0]
        if ip_address in running_algorithms:
            return {'status': False, 'error': 'Min max has already started on that camera'}
        directory = f'images/{ip_address}'
        print(directory, 'directory')
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as e:
        print('folder doesnt created', e)
        return {'status': False, 'error': 'Folder doesnt created'}
    try:
        algData = algorithms[data.algorithm]
    except:
        print('Algorithm not found', data)
        return {'status': False, 'error': 'Algorithm not found'}
    try:
        file = algData["file"]
        box_model_weights = algData['box_model']
        human_model_weights = algData['human_model']
        function = algData['algorithm']
        server_url = data.server_url
        print(data, 'data')
        print(data.camera_url)
        if data.algorithm == 'min_max_control':
            print(data.server_url)
            areas = data.extra
        else:
            areas = ''
        if data.algorithm == 'min_max_control' and not areas[0]["itemId"]:
            print('Algorithm not running')
            return {'status': False, 'error': 'Algorithm not running, check fields'}
        ip_address_server = re.findall(r'(?:\d{1,3}\.)+(?:\d{1,3})', server_url)[0]
        source = f'http://{ip_address}/onvif-http/snapshot?Profile_1'
        if ip_address_server == ip_address:
            source = f'http://{ip_address}:3456/onvif-http/snapshot?Profile_1'
        areas = json.dumps(areas)
        print(areas, 'areas', type(areas))
        command = ["python", "-u", file, "--box_model", box_model_weights, "--nosave", "--source",
                   source, "--device", "cpu", "--classes", "0", "--server_url", ip_address_server, "--ip_address",
                   ip_address, "--algorithm", data.algorithm, '--username', 'admin', '--password', 'just4Taqtile',
                   '--areas', areas,
                   "--human_detect_model", human_model_weights]
        task = asyncio.create_task(function(areas, 'admin', 'just4Taqtile', [data.algorithm], [ip_address], [ip_address_server], source, [box_model_weights], False, False, 640, [human_model_weights], 5))
        pid = random.randint(0, 99999999)
        pids[str(pid)] = task
        cameras[str(pid)] = ip_address
        # running_algorithms.append(ip_address)
        return {'status': True, 'pid': pid}
    except Exception as e:
        print(e, 'run error')
        return {'status': False}
    return {'status': False}


@app.post("/stop", status_code=201)
async def stop(data: StopData):
    global pids
    global running_algorithms
    global cameras
    print('stop alg pid:', str(data.pid))
    print('camera', cameras[str(data.pid)])
    try:
        task = pids[str(data.pid)]
        if task:
            task.cancel()
            print("Task stopped")
            pids[str(data.pid)] = None
            if cameras[str(data.pid)] in running_algorithms:
                running_algorithms.remove(cameras[str(data.pid)])
        return {'status': True}
    except Exception as e:
        print(e, 'stop error')
        return {'status': False}

@app.post("/info", status_code=201)
async def info():
    return [
        {
            "name": "Idle Control Python",
            "version": "0.3.5"
        },
        {
            "name": "MinMax Control Python",
            "version": "0.3.5"
        },
    ]

