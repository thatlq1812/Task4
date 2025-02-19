@echo off
setlocal

call conda activate torch

cd D:\OJT\Task4

docker stop triton_server triton_client
