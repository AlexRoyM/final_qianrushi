import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights  # 导入 ResNet18权重
import os
from PIL import Image
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.utils.class_weight import compute_class_weight
from torchvision.transforms import ToTensor, ToPILImage
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_metrics(loader, model, device):
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    return precision, recall, f1


def add_noise(img):
    # 将PIL图像转换为Tensor
    img_tensor = ToTensor()(img)

    # 创建与图像尺寸相同的噪声张量
    noise = torch.randn(img_tensor.size()) * 0.1

    # 向图像添加噪声
    noisy_img_tensor = img_tensor + noise

    # 如果需要，将Tensor转换回PIL图像
    noisy_img = ToPILImage()(noisy_img_tensor.clamp(0, 1))

    return noisy_img


class GestureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # GestureDataset类的构造函数
        self.root_dir = root_dir  # 存储存放图片的根目录
        self.transform = transform  # 存储要应用于图像的转换函数
        # 初始化列表以存储图像文件路径及其对应的标签
        self.images = []
        self.labels = []
        # 定义有效的图像文件扩展名
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        for label in range(7):  # 遍历3个类别/标签
            # 为每个标签构建文件夹路径
            label_folder = os.path.join(self.root_dir, str(label))
            # 遍历标签文件夹中的所有文件
            for img_file in os.listdir(label_folder):
                # 检查文件是否具有有效的图像文件扩展名
                if os.path.splitext(img_file)[1].lower() in valid_extensions:
                    # 将图像文件的完整路径添加到images列表中
                    self.images.append(os.path.join(label_folder, img_file))
                    # 将图像的标签添加到labels列表中
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('L')  # 转为灰度图
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class GestureCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # 加载预训练的 ResNet-18 模型
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 添加Dropout层
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.dropout5 = nn.Dropout(0.5)

        # 替换最后的全连接层以适应新的分类任务
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # 通过ResNet模型，不包括最后的全连接层
        x = self.model.conv1(x)
        x = self.dropout1(x)  # 第一个Dropout层
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.dropout2(self.model.layer1(x))  # 在layer1后添加Dropout
        x = self.dropout3(self.model.layer2(x))  # 在layer2后添加Dropout
        x = self.model.layer3(x)
        x = self.dropout4(self.model.layer4(x))

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        # 应用Dropout
        x = self.dropout5(x)

        # 通过修改后的全连接层
        return self.model.fc(x)


def calculate_accuracy(loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train_model(model, train_loader, val_loader, device, num_epochs=100, early_stopping_patience=5):
    labels = [label for _, label in train_loader.dataset]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # 冻结 layer1 和 layer2
    for name, param in model.named_parameters():
        param.requires_grad = True
        # if "layer1" in name or "layer2" in name:
        #     param.requires_grad = False
        # else:
        #     param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.01)  # 添加 weight_decay 参数
    best_val_accuracy = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_accuracy = calculate_accuracy(val_loader, model, device)
        precision, recall, f1 = calculate_metrics(val_loader, model, device)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Val Accuracy: {val_accuracy}%, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

        # 早停逻辑
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    return model


def predict_and_plot_examples(model, dataset, device, num_images=30):
    # 确保模型处于评估模式
    model.eval()

    # 随机选择图片
    indices = np.random.choice(len(dataset), num_images, replace=False)
    images, labels = zip(*[dataset[i] for i in indices])

    # 进行预测
    with torch.no_grad():
        images_tensor = torch.stack(images).to(device)
        outputs = model(images_tensor)
        _, predicted = torch.max(outputs, 1)

    # 绘制图片和预测标签
    plt.figure(figsize=(20, 10))
    for i in range(num_images):
        plt.subplot(2, num_images // 2, i + 1)
        # 将单通道图像重复三次以形成三通道图像
        img = images[i].repeat(3, 1, 1).permute(1, 2, 0)
        plt.imshow(img, cmap='gray')  # 使用灰度色彩映射
        plt.title(f'{predicted[i].item()}')
        plt.axis('off')
    plt.show()


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # 新增随机水平翻转
        transforms.RandomRotation(10),  # 新增随机旋转
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.Lambda(add_noise),
        transforms.ToTensor()
    ])
    dataset = GestureDataset('../gesture_pic/train_data', transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureCNN().to(device)

    trained_model = train_model(model, train_loader, val_loader, device)

    # 保存模型的状态字典
    torch.save(trained_model.state_dict(), '../model/resnet18_2.pth')

    # 使用此函数进行预测和绘图
    predict_and_plot_examples(model, dataset, device)


if __name__ == '__main__':
    main()
