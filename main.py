from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image
import tritonclient.http as httpclient
import numpy as np
import io

# Create FastAPI app

image_classification_app = FastAPI()

# Add CORS middleware
image_classification_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Triton Server connection variables
TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "densenet_onnx"
INPUT_NAME = "data_0"
OUTPUT_NAME = "fc6_1"

# Connect to Triton Server
client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

# Function for image preprocessing
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transformed_img = transform(image).numpy()
    return transformed_img

# Define the endpoint for inference
@image_classification_app.post("/inference")
async def inference(file: UploadFile = File(...)):
    # Read the image file from the request
    image_bytes = await file.read()
    input_data = preprocess_image(image_bytes)

    # Create request to Triton
    inputs = httpclient.InferInput(INPUT_NAME,
                                   input_data.shape, 
                                   datatype="FP32")
    inputs.set_data_from_numpy(input_data, binary_data=True)

    # Define the output from Triton
    outputs = httpclient.InferRequestedOutput(OUTPUT_NAME, 
                                              binary_data=True, 
                                              class_count=1000)
    
    # Send the request to Triton
    results = client.infer(model_name=MODEL_NAME, 
                           inputs=[inputs], 
                           outputs=[outputs])
    
    inference_output = results.as_numpy(OUTPUT_NAME).astype(str)

    # Get the top 5 predictions
    output = np.squeeze(inference_output)[:5]

    predictions = []
    for i, line in enumerate(output):
        parts = line.split(":")
        if len(parts) < 3:
            predictions.append(f"Cannot process the line, `{line}`")
            continue

        prob = round(float(parts[0]), 2)
        result = parts[2].strip()
        predictions.append({"label": result, "probability": prob})
    return {"predictions": predictions}



