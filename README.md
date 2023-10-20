# About Min-Max
Min-Max is one of the Official [5controlS](https://5controls.com/) algorithm.

With Min-Max, you can now effortlessly manage your reserves and ensure smooth and uninterrupted production processes. Our software helps you maintain the optimal stock levels by providing real-time inventory tracking and analysis, allowing you to make informed decisions and avoid costly production line stoppages. 

![image](https://github.com/5sControl/min-max/assets/131950264/3751506b-12ba-4455-92da-8382421ec4c8)

## Key features

- stock levels monitoring;
- notifications about critical levels;
- automatic ordering for just-in-time deliveries.

**Plug-in Min-Max to 5controlS platform to start monitoring your inventory level!**

> Learn more about Operation control on the [5controlS website](https://5controls.com/solutions/inventory-control).

# Getting started 

### Build image for min_max_python algorithm
- For x86 users

    ```docker build -t 5scontrol/min_max_python:latest .```

- for AArch64 users 

    ```docker buildx build --platform linux/amd64 -t 5scontrol/min_max_python:latest .```


### Build image for min_max_python-server algorithm

- For x86 users

    ```docker build -t 5scontrol/min_max_python-server:latest ./model_image```

- For AArch64 users 

    ```docker build buildx --platform linux/amd64 -t 5scontrol/min_max_python-server:latest ./model_image```



### Run containers

*Check id of container:* ```docker image list```

- For min_max_python

    ```docker run -rm -it min_max_python -e <variables>```

- For min_max_python-server

    ```docker run -rm -it min_max_python-server```

  To run min_max algorithm you have to pass following variables:
    - ```folder``` -- folder for saving images
    - ```camera_url``` -- camera url
    - ```server_url``` -- server url
    - ```extra``` -- areas data


### Run/Test code

- For min_max_python

  ```python main.py```

- For min_max_python-server

  ```cd ./model_image && python -m flask run --host 0.0.0.0 --port 5000```


### Push images

- For min_max_python:

  ```docker image push 5scontrol/min_max_python:latest```

- For min max server image:

  ```docker image push 5scontrol/min_max_python-server:latest```

# **Documentation**

[Documentation for Developers](https://github.com/5sControl/5s-dev-documentation/wiki)

[User Documentation](https://github.com/5sControl/Manufacturing-Automatization-Enterprise/wiki)


# **Contributing**
Thank you for considering contributing to 5controlS. We truly believe that we can build an outstanding product together!

We welcome a variety of ways to contribute. Read below to learn how you can take part in improving 5controlS.

## **Code of conduct**

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## Code contributing

If you want to contribute, read  our [contributing guide](CONTRIBUTING.md) to learn about our development process and pull requests workflow.

We also have a list of [good first issues](https://github.com/5sControl/min-max/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) that will help you make your first step to beсoming a 5S contributor.


# **Project repositories**

**5controlS Platform:**
1. [5s-webserver](https://github.com/5sControl/5s-webserver)
2. [5s-backend](https://github.com/5sControl/5s-backend)
3. [5s-frontend](https://github.com/5sControl/5s-frontend)
4. [5s-algorithms-controller](https://github.com/5sControl/5s-algorithms-controller)
5. [5s-onvif](https://github.com/5sControl/5s-onvif)
6. [5s-onvif-finder](https://github.com/5sControl/5s-onvif-finder)
   
**Official Algorithms:**
1. [min-max](https://github.com/5sControl/min-max)
2. [idle-control](https://github.com/5sControl/idle-control)
3. [operation-control-js](https://github.com/5sControl/operation-control-js)
4. [machine-control](https://github.com/5sControl/machine-control)
5. [machine-control-js](https://github.com/5sControl/machine-control-js)

**Algorithms Servers:**
1. [inference-server-js](https://github.com/5sControl/inference-server-js)


# **License**
[AGPL-3.0](LICENSE)

> Min-Max uses third party libraries that are distributed under their own terms (see [LICENSE-3RD-PARTY.md](https://github.com/5sControl/min-max/blob/main/LICENSE-3RD-PARTY.md)).<br>

<br>
<div align="center">
  <a href="https://5controls.com/" style="text-decoration:none;">
    <img src="https://github.com/5sControl/Manufacturing-Automatization-Enterprise/blob/3bafa5805821a34e8b825df7cc78e00543fd7a58/assets/Property%201%3DVariant4.png" width="10%" alt="" /></a> 
  <img src="https://github.com/5sControl/5s-backend/assets/131950264/d48bcf5c-8aa6-42c4-a47d-5548ae23940d" width="3%" alt="" />
  <a href="https://github.com/5sControl" style="text-decoration:none;">
    <img src="https://github.com/5sControl/Manufacturing-Automatization-Enterprise/blob/3bafa5805821a34e8b825df7cc78e00543fd7a58/assets/github.png" width="4%" alt="" /></a>
  <img src="https://github.com/5sControl/5s-backend/assets/131950264/d48bcf5c-8aa6-42c4-a47d-5548ae23940d" width="3%" alt="" />
  <a href="https://www.youtube.com/@5scontrol" style="text-decoration:none;">
    <img src="https://github.com/5sControl/Manufacturing-Automatization-Enterprise/blob/ebf176c81fdb62d81b2555cb6228adc074f60be0/assets/youtube%20(1).png" width="5%" alt="" /></a>
</div>

