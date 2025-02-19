@echo off

call conda activate torch

cd D:\OJT\Task4

docker exec triton_client perf_analyzer -m densenet_onnx -u localhost:8000

pause
