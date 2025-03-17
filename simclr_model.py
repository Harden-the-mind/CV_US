import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class SimCLRTransforms:
    """SimCLR使用的数据增强"""
    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

class LiverUltrasoundDataset(Dataset):
    """超声图像数据集"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # 遍历所有医院文件夹
        hospitals = ['中山', '十院', '同济']
        for hospital in hospitals:
            hospital_path = os.path.join(root_dir, hospital)
            if not os.path.exists(hospital_path):
                continue
                
            # 遍历病例文件夹
            for case in os.listdir(hospital_path):
                case_path = os.path.join(hospital_path, case)
                if not os.path.isdir(case_path):
                    continue
                    
                # 获取.tif文件
                for img_file in os.listdir(case_path):
                    if img_file.endswith('.tif'):
                        self.image_paths.append(os.path.join(case_path, img_file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img1, img2 = self.transform(image)
            return img1, img2
        return image

class SimCLRModel(nn.Module):
    """SimCLR模型"""
    def __init__(self, feature_dim=128):
        super(SimCLRModel, self).__init__()
        
        # 使用MobileNetV2作为基础编码器
        self.encoder = models.mobilenet_v2(pretrained=True)
        self.feature_dim = feature_dim
        
        # 获取最后一层的输入维度
        last_channel = self.encoder.last_channel
        
        # 修改分类器
        self.encoder.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class SimCLR:
    """SimCLR训练框架"""
    def __init__(self, model, optimizer, temperature=0.5):
        self.model = model
        self.optimizer = optimizer
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def info_nce_loss(self, features):
        batch_size = features.shape[0] // 2
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        
        # 去除对角线上的自身相似度
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # 计算对比损失
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)
        
        logits = logits / self.temperature
        return self.criterion(logits, labels)

def train_simclr():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据集和数据加载器
    dataset = LiverUltrasoundDataset(
        root_dir='三星图像 三个中心/三星图像 三个中心',
        transform=SimCLRTransforms()
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # 创建模型和优化器
    model = SimCLRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    simclr = SimCLR(model, optimizer)

    # 训练循环
    num_epochs = 50  # 减少训练轮数
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (img1, img2) in enumerate(dataloader):
            images = torch.cat([img1, img2], dim=0).to(device)
            features = model(images)
            loss = simclr.info_nce_loss(features)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch}/{num_epochs}] Average Loss: {avg_loss:.4f}')
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, f'simclr_checkpoint_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train_simclr() 