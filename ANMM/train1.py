import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet


def main():
    torch.cuda.empty_cache()  # 释放未使用的显存

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 增强的数据转换策略
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(180, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize(200),
            transforms.CenterCrop(180),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    # 最高用的train1,danzao15(1)
    data_root = r'D:/team/alexnet learning/deep-learning-for-image-processing-master-juanji-msa/data_set/danzao15(1)'
    train_dataset = datasets.ImageFolder(os.path.join(data_root, 'train'), data_transform['train'])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    with open('class_indices.json', 'w') as json_file:
        json.dump(cla_dict, json_file, indent=4)

    # 调整批量大小
    batch_size = 64
    # 32 0.0001 1e-3 0.987
    # 64 0.0001 1e-3 0.9891(目前：2025.11.11)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 增加工作线程数
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw, pin_memory=True)

    validate_dataset = datasets.ImageFolder(os.path.join(data_root, 'val'), data_transform['val'])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, pin_memory=True)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # 改进的模型初始化
    net = AlexNet(num_classes=5, init_weights=True)
    net.to(device)
    print(net)

    # 使用带权重衰减的优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.0001, weight_decay=1e-3)  # weight_decay=1e-3，1e-2

    # 改进的学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=300)

    # 增强的早停机制
    epochs = 300
    patience = 300  # 增加耐心值
    best_acc = 0.0
    best_loss = float('inf')
    counter_no_improve = 0
    early_stop = False

    save_path = './ANMM.pth'
    results_file = 'training_results.csv'

    with open(results_file, 'w') as f:
        f.write('Epoch,Train Loss,Train Acc,Val Loss,Val Acc,Learning Rate\n')

    for epoch in range(epochs):
        if early_stop:
            print("Early stopping triggered!")
            break

        # 训练阶段
        net.train()
        running_loss = 0.0
        train_correct = 0

        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # 更新学习率

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.4f} lr:{:.6f}".format(
                epoch + 1, epochs, loss, optimizer.param_groups[0]['lr'])

        train_loss = running_loss / train_num
        train_acc = train_correct / train_num

        # 验证阶段
        net.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                outputs = net(val_images)
                loss = loss_function(outputs, val_labels)

                val_loss += loss.item() * val_images.size(0)
                predict_y = torch.max(outputs, dim=1)[1]
                val_correct += (predict_y == val_labels).sum().item()

        val_loss = val_loss / val_num
        val_acc = val_correct / val_num

        current_lr = optimizer.param_groups[0]['lr']
        print('[epoch %d] train_loss: %.4f train_acc: %.4f val_loss: %.4f val_accuracy: %.4f lr: %.6f' %
              (epoch + 1, train_loss, train_acc, val_loss, val_acc, current_lr))

        # 记录训练结果
        with open(results_file, 'a') as f:
            f.write(f'{epoch + 1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{current_lr:.6f}\n')

        # 改进的早停机制：同时考虑准确率和损失
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            if val_acc > best_acc:
                best_acc = val_acc
            if val_loss < best_loss:
                best_loss = val_loss
            torch.save(net.state_dict(), save_path)
            counter_no_improve = 0
        else:
            counter_no_improve += 1
            print(f'Validation metrics did not improve for {counter_no_improve}/{patience} epochs')

            if counter_no_improve >= patience:
                early_stop = True
                print(
                    f"Early stopping after {epoch + 1} epochs with best acc: {best_acc:.4f} and best loss: {best_loss:.4f}")

    print('Finished Training')


if __name__ == '__main__':
    main()
