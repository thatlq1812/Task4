@echo off
setlocal

call conda activate torch

cd D:\OJT\Task4

docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 --name triton_server -v D:\OJT\Task4\model_repository:/models nvcr.io/nvidia/tritonserver:25.01-py3 tritonserver --model-repository=/models

pause
