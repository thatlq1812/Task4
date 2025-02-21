import requests

url = "http://localhost:8080/inference"
image_path = "img1.jpg"

with open(image_path, "rb") as img_file:
    files = {"file": img_file}
    response = requests.post(url, files=files)

print(response.json())  # Prediction results
