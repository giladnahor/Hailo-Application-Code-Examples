#!/bin/bash

# download hef files to ./resources

wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.9.0/clip_resnet_50x4.hef -P ./resources
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/personface_detector/yolov5s_personface/2022-04-01/yolov5s_personface.hef -P ./resources