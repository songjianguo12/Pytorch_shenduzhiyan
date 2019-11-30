import torch
from DogCat_dataset import DogCatDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

path_model = "./3_best.pkl"
net = torch.load(path_model)
# net = torch.load(path_model, map_location="cpu")

test_dir = "./test_data"

test_data = DogCatDataset(data_dir=test_dir, transform=valid_transform)
valid_loader = DataLoader(dataset=test_data, batch_size=1)

for i, data in enumerate(valid_loader):
    # print(valid_loader)
    # forward
    inputs, labels = data
    # print(inputs, labels) #返回的是图片的像素矩阵 和label
    outputs = net(inputs.cuda())
    # print(outputs.data)
    _, predicted = torch.max(outputs.data, 1)#1代表维度
    # print(predicted.cpu().numpy()[0])#转化成numpy取出对应标签

    """
    rmb_label = {"dog": 0, "cat": 1}
    """

    animal = "dog" if predicted.cpu().numpy()[0] == 0 else "cat"
    print("是：{}".format(animal))

