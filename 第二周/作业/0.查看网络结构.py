from lenet import LeNet, MyNet
from torchsummary import summary


net = MyNet(classes=2)
net.initialize_weights()

summary(net, input_size=(3, 40, 40),device='cpu')

# print(net)