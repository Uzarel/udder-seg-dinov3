# udder-seg-dinov3

This repository contains the code release for the scientific paper **Lightweight Thermal Udder Segmentation via Structured Pruning for On-Device Farm Deployment**. It includes training scripts, as well as an inference module designed for NVIDIA Jetson devices, exposed as a REST API.

## Table of Contents

1. [Training](#training)
2. [Configuration File](#configuration-file)
3. [Inference on NVIDIA Jetson](#inference-on-nvidia-jetson)
4. [Training Logs and Model Weights](#official-training-logs-and-model-weights)

## Training

The [`train+prune.py`](train+prune.py) script is the main entry point for training the model. This script trains the model on your dataset, applies pruning techniques to optimize the model for deployment, and finally fine-tunes it.

### Prerequisites

Ensure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Script

To run the training process, execute the following command:

```bash
python train+prune.py
```

### Key Parameters

The script uses several parameters defined in the [`config.ini`](config.ini) file. These parameters control the training process, as well as other things like logging.

## Configuration File

Below is a snippet of the structure of the file:

```ini
[TRAINING]
batch_size = 32
learning_rate = 0.001
epochs = 50
```

- **`batch_size`**: Number of samples per batch during training.
- **`learning_rate`**: Learning rate for the optimizer.
- **`epochs`**: Total number of training epochs.

Modify these values and the others as needed to suit your dataset and hardware.

## Inference on NVIDIA Jetson

The [`jetson`](jetson/) folder contains the code for deploying the model on NVIDIA Jetson devices. The model is exposed as a REST API for inference. Note that in this release, we specifically address the Nvidia Jetson Orin Nano devices, but it can be easily changed to address other Jetson devices.

### Folder Structure

- [`app/serve_cuda.py`](jetson/app/serve_cuda.py): The main script for running the REST API server.
- [`client.py`](jetson/client.py): A client script to send inference requests to the server for testing purposes.
- [`models/`](jetson/models/): The location of your ONNX model for inference.
- [`Dockerfile`](jetson/Dockerfile): Docker configuration for building and deploying the application.

### Running the Inference Server

1. Place your ONNX model in the [`jetson/models/`](jetson/models/) folder.
2. Move to the `jetson` folder.
2. Build the Docker image:

   ```bash
   docker build -t jetson-inference .
   ```

3. Run the Docker container:

   ```bash
   docker run -d \
   --runtime nvidia \
   --network host \
   -e MODEL_PATH=/models/your_model.onnx \
   -e WARMUP_RUNS=5 \
   -v $(pwd)/models:/models:ro \
   jetson-inference
   ```

4. The REST API segmentation endpoint will be available at `http://<JETSON_IP>:8000/segment`.

### Sending Inference Requests

For testing, you can use the provided [`client.py`](jetson/client.py) script to send inference requests:

```bash
python client.py --url http://<JETSON_IP>:8000/segment --image path/to/image.png
```

We also provide an example image [`example.png`](jetson/example.png) for this purpose.

## Official Training Logs and Model Weights

- **Training Logs**: [WandB](https://wandb.ai/mat094/udder-seg-dinov3)
- **Model Weights**: [Google Drive](https://drive.google.com/file/d/1QEOItHVCtlUNCP4_7b4F2QHku94Yb1fI/view?usp=sharing)

This repository is released under the MIT [`LICENSE`](LICENSE). For any questions, please contact the authors of the paper or open an issue here on the official repository.
