#!/bin/bash
# an example to start a SatSim container and mount the source code (this directory) into /workspace
# run "make develop" in the container in the /workspace directory to setup the developer environment
docker run -d --name satsim_dev --runtime=nvidia --privileged -p 6006:6006 -p 8888:8888 -it -v $(pwd):/workspace -v /home/$USER/share:/workspace/share satsim
sleep 5
docker logs satsim_dev