from PIL import Image
from torchvision import transforms
import Net
import torch

net_test = Net.Net()
net_test.load_state_dict(torch.load('../Result/Net_result_2022_06_05T15_49_10.pth'))

path = '../Data/DataSet/T1/4/T1_4_109.jpg'
img = Image.open(path)
print("图像的路径为{}".format(path))

transform = transforms.ToTensor()
img = transform(img)
img = torch.reshape(img, [-1, 1, 16, 16])
outs = net_test(img)
out_index = outs.argmax(1)
if out_index == 0:
    print("故障类别为1")
elif out_index == 1:
    print("故障类别为4")
elif out_index == 2:
    print("故障类别为7")
elif out_index == 3:
    print("故障类别为8")
else:
    print("无法判断")
