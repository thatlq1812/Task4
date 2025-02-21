@echo off
setlocal

call conda activate torch

cd D:\OJT\Task4

uvicorn main:image_classification_app --host 0.0.0.0 --port 8080 --reload

pause
