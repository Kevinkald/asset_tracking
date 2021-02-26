# TensorFlow Object Detection on Docker

These instructions are experimental.

## Building and running:

```bash
# From the root of the git repository. Assuming this command is ran from models inside Tensorflow dir.
docker sudo docker build -f ../asset_tracking/docker/Dockerfile -t od .
docker run -it od
```
