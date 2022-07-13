import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import Net
import Readdata


train_data_set = Readdata.train_dataset
test_data_set = Readdata.test_dataset

train_data_size = len(train_data_set)
test_data_size = len(test_data_set)

print('========================================')
print('训练集的长度为：{}'.format(train_data_size))
print('测试集的长度为：{}'.format(test_data_size))
print('========================================')

train_data_loader = DataLoader(train_data_set, batch_size=1, shuffle=True)
test_data_loader = DataLoader(test_data_set, batch_size=1, shuffle=True)

learning_rate = 0.01
epochs = 10
total_train_step = 0
total_test_step = 0
time = "{0:%Y_%m_%dT%H_%M_%S}".format(datetime.now())

net = Net.Net()
net.cuda()

loss_fc = nn.CrossEntropyLoss()
loss_fc.cuda()
optimizer = torch.optim.SGD(params=net.parameters(), lr=learning_rate)
writer = SummaryWriter('logs/{}/'.format(time))

for epoch in range(epochs):
    print("-------第{}轮训练开始--------".format(epoch + 1))

    for data in train_data_loader:
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.cuda()
        outs = net(imgs)
        loss = loss_fc(outs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        writer.add_scalar('train_loss', loss, total_train_step)
        if total_train_step % 200 == 0:
            print("训练次数:{}, loss:{}".format(total_train_step, loss.item()))

    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data_loader:
            imgs, labels = data
            imgs = imgs.cuda()
            labels = labels.cuda()
            outs = net(imgs)
            loss = loss_fc(outs, labels)
            total_test_loss += loss.item()
            accuracy = (outs.argmax(1) == labels).sum()
            total_accuracy += accuracy
    total_test_step += 1
    print("第{}轮训练测试集loss:{}".format(epoch + 1, total_test_loss))
    print("第{}轮训练测试集正确率:{}".format(epoch + 1, total_accuracy / test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('accuracy', total_accuracy / test_data_size, total_test_step)

torch.save(net.state_dict(), '../Result/Net_result_{}.pth'.format(time))
print("模型已保存")
writer.close()
