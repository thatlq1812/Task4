import tritonclient.http as httpclient
import numpy as np
from torchvision import transforms
from PIL import Image
import IPython.display as display
from PIL import Image

image = Image.open("img1.jpg") ######## Change the path to the image you want to test
display.display(image)

# Resize and normalize the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
img = Image.open("img1.jpg")
transformed_img = transform(img).numpy()  # Shape: [3, 224, 224]

# Connect to the server
client = httpclient.InferenceServerClient(url="localhost:8000")

# Create the input and output
inputs = httpclient.InferInput("data_0", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput("fc6_1", binary_data=True, class_count=1000)

# Send the request
results = client.infer(model_name="densenet_onnx", inputs=[inputs], outputs=[outputs])
inference_output = results.as_numpy('fc6_1').astype(str)

# Show the top 5 classes
output = np.squeeze(inference_output)[:5]

print("Top 5 classes:")
for i, prob in enumerate(output):
    print(f"{i+1}: {prob}")