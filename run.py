import torch

from model import PaCaVIT

batch_size = 17
img_size = 224

x = torch.rand(batch_size, 3, img_size, img_size)
model = PaCaVIT()

output = model(x)
print('output shape:', output.shape)

