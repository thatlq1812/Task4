
@echo off
setlocal

call conda activate torch

cd D:\OJT\Task4

docker run -it --rm --name triton_client --net=host nvcr.io/nvidia/tritonserver:25.01-py3-sdk 

pause
