#!/bin/bash

sudo docker stop ubermejo-activity-recognition
sudo docker rm ubermejo-activity-recognition
sudo docker run -it --ipc=host --gpus "device=0" --name ubermejo-activity-recognition -v activity-recognition:/results ubermejo/activity_recognition bin/bash
