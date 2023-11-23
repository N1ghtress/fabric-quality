# Machine Learning 1 Project 2023 

Students:
Marc Brustolin p1605812
Victor Dupriez p1816740
Victor Faure p2108009

## Predict images ASAP
This section is for those who want to test the app as fast as possible. The next steps will lead you through downloading the right files and executing the right commands to start our app.

### Clone repository
```sh
git clone git@forge.univ-lyon1.fr:p2108009/ml_projet.git
```

### Download models and test data
Here you can download models, ref_data.zip and test images.

[models archive](https://www.mediafire.com/file/spkv5aj4bwll6em/models.zip/file)

[ref_data.zip](https://www.mediafire.com/file/qy84ptoqltzg7jt/ref_data.zip/file)

[images archive](https://www.mediafire.com/file/xbhewm196vc0lgx/test.zip/file)

In order for the app to work, you must unzip the models and move the `artifacts/` directory to repository root directory. Then move `ref_data.zip` to `data/` directory. Finally unzip the `test.zip` archive in order to select them from frontend file uploader.

### Start services
From here you should be able to start frontend and backend using `docker compose`, such as shown.

```sh
docker compose -f webapp/docker-compose.yml up &
docker compose -f serving/docker-compose.yml up &
```

### Predict images & Give feedback
If all went well, you should be able to connect to the [frontend](http://0.0.0.0:8081/) after some time. From there, you can upload images, get a prediction and send feedback to the app about that prediction.

One thing to note is that the `good` class is coded with 1.

## From scratch usage
In this section you will follow through all the data preprocessing we did to obtain `ref_data.csv` and train the model on your machine.

### Clone repository

### Download data and models

### Move archive

### Create ref_data.csv

### Start services
