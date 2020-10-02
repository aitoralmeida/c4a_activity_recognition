#!/bin/bash

sudo docker stop ubermejo-activity-recognition-2
sudo docker rm ubermejo-activity-recognition-2
sudo docker run -it --ipc=host --gpus "device=0" --name ubermejo-activity-recognition-2 -v activity-recognition:/results ubermejo/activity_recognition bin/bash
