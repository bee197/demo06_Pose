import torch
from ultralytics import YOLO
from LSTM.model import lstm as LSTM

model = LSTM()
model.load('models/lstm.pth')
input = torch.randn(32, 5, 39)
input_names = ['input_0']
output = model(input)
output_names = ['output']
torch.onnx.export(model, input, 'lstm.onnx',
                  input_names=input_names,
                  output_names=output_names,
                  opset_version=11)
# Load a model
# model = YOLO("lstm.pt")  # load a pretrained model (recommended for training)
# success = model.export(format="onnx", simplify=True)  # export the model to onnx format
# assert success