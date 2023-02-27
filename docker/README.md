SatSim Docker
=============

Docker must be installed to build and run the SatSim Docker container. To run the SatSim Docker container, the Nvidia Docker runtime must be installed and the host machine configured to run CUDA. Choose the Docker build file that matches the host computer CUDA version.

Building
--------

To create a Docker container, run `make docker` from the root SatSim project folder.

Alternatively, run this command in this folder:

```
docker build -t satsim:latest -f ubuntu20.04_cuda11.2_py3.8.dockerfile ..
```

`make dist` and `make docs` must be run from the root SatSim project folder before running `docker build`.


Running
-------

Run the following command to run Jupyter Lab in the SatSim docker container:

```
docker run --runtime=nvidia -p 8888:8888 satsim
```

Then open a web browser to `http://localhost:8888` to access Jupyter Lab interface.

To mount a local volume to read and write data to, use the `-v` option. The following example mounts the current directory to `/workspace/share`:

```
docker run --runtime=nvidia -p 8888:8888 -v $(pwd):/workspace/share satsim
```

Alternatively, SatSim can be run directly. For example, to bind the current directory into the container and run SatSim in that folder:

```
docker run --runtime=nvidia -v $(pwd):/workspace/share -w /workspace/share satsim satsim --debug INFO run --device 0 --output_dir ./images my_satsim_config.json
```
