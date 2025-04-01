# Trading API

### Abstract
This repo is a showcase of backend orchistration. Containing a REST API (FastAPI), distributed task queue (Celery), visualizer (JS served w/ FastAPI), and a basic yet very overconfident AI/ML model (Tensorflow). 

---

***NOTE*** The model is most definitely terrible. I am not claiming to be an expert in AI/ML, nevertheless I am using it to showcase my skills in applying algorithms in a useful and meaningful way. 

---

### Component Layout

- FastAPI: My prefered python web framework. I find it easy to stand up and has type validation out of the box with Pydantic. There are 2 main endpoints. `POST: /api/v1/stock/${symbol}` and `GET: /api/v1/stock/${symbol}`. The POST request will generate the model and a plot for a specific stock symbol. The GET request will get the model and performance graphic for that stock. 
- Celery: I am on the fence with using Celery. Understandably it is the most popular distributed task queue. I just dont like the abstraction layer and bloat it contains (sometimes I dont want to use a queue). I will use it because it is an industry standard and admidatly I find it easy to use. This is where the main AI/ML logic lives. Upon getting a POST request, the API server will send out a "celery task request" that will run the model training and produce the perfromance graphic.
- Mongo: Storage for the model and graphics. I like GridFS and mongo is really simple to set up.
- Rabbitmq: The queue for celery, no specific reason for using it, just what I am used to.
--- 

### Containerization

A `Dockerfile` is provided. This will build a container with a python environment. I do not have a CI/CD pipeline set up to auto-build the containers and push to a registry due to cost reasons (this is supposed to be a fun side project).

To run the system in docker, use the `docker-compose.yaml` file to run in docker. 

***NOTE*** You need to set environment variables for your [Alpaca](https://alpaca.markets) account. The api reads in the `API_KEY` and `SECRET` environment variables which can be found in the `src/trading_api/env.py` file.


```bash
docker-compose up
```

To run the system in kubernetes, `cd` into the helm directory and run:

```bash
helm install trading-api .
```

To access the api you must run 

```bash
kubectl port-forward <pod for trading-api> 8000:8000 -n trading_api
```

The image is quite large because its running a development python virtual environment. Ideally in production this virtual environment would not be there.