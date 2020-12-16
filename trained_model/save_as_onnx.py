import onnx
import torch
import torch.nn as nn
import numpy as np
import lstm_class


#===========================================#
#        Saves Model as .onnx File          #
#===========================================#

net = torch.load('trained_model.pt')
net.eval()

with torch.no_grad():
    input = torch.tensor([[1,2,3,4,5,6,7,8,9]])
    print(input.dtype)
    h0, c0 = net.init_hidden(1)
    print(h0.dtype)
    print(c0.dtype)
    output, (h1, c1) = net.forward(input, (h0,c0))


    torch.onnx.export(net, (input, (h0, c0)), 'trained_model.onnx',
                    input_names=['input', 'h0', 'c0'],
                    output_names=['output', 'h1', 'c1'],
                    dynamic_axes={'input': {0: 'sequence'}})

    onnx_model = onnx.load('trained_model.onnx')
