import torchvision.models as models
import pickle
import torch
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import numpy as np

model = models.resnet50(num_classes=8631,pretrained=False)
with open('../models/resnet50_ft_weight.pkl', 'rb') as f:
    obj = f.read()

weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj).items()}
model.load_state_dict(weights)

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

model.eval()

imagepath = '../data/Images/img_3609.bmp'
image = Image.open(imagepath)
imgblob = Variable(normalize(to_tensor(scaler(image))).unsqueeze(0))

tf_last_layer_chopped = torch.nn.Sequential(*list(model.children())[:-1])
output = tf_last_layer_chopped(imgblob)
print(np.shape(output))
