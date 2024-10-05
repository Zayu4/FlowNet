import torch
import torch.nn as nn
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化資料集
        :param root_dir: 圖片所在的主目錄 (e.g., 'Dataset/Training/')
        :param transform: 圖像的轉換 (e.g., ToTensor, Resize等)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # 遍歷資料夾並收集所有 png 圖片的路徑
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    if img_name.endswith("*.png"):
                        img_path = os.path.join(folder_path, img_name)
                        self.image_paths.append(img_path)

    def __len__(self):
        # 回傳資料集中圖片的數量
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 根據索引 idx 加載圖片
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # 如果有轉換操作，則進行轉換
        if self.transform:
            image = self.transform(image)
        
        return image

class FlowNetSimple(nn.Module):
    def __init__(self):
        super(FlowNetSimple, self).__init__()
        # 定義卷積層
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        
        # 定義反卷積層 (上採樣層)
        self.deconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        
        # 光流預測層
        self.predict_flow6 = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        self.predict_flow5 = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.predict_flow4 = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.predict_flow3 = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)
        self.predict_flow2 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 下採樣部分 (Encoder)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv3_1 = self.conv3_1(conv3)
        conv4 = self.conv4(conv3_1)
        conv4_1 = self.conv4_1(conv4)
        conv5 = self.conv5(conv4_1)
        conv5_1 = self.conv5_1(conv5)
        conv6 = self.conv6(conv5_1)
        
        # 上採樣部分 (Decoder)
        deconv5 = self.deconv5(conv6)
        deconv4 = self.deconv4(torch.cat((deconv5, conv5_1), 1))
        deconv3 = self.deconv3(torch.cat((deconv4, conv4_1), 1))
        deconv2 = self.deconv2(torch.cat((deconv3, conv3_1), 1))
        
        # 光流預測
        flow6 = self.predict_flow6(conv6)
        flow5 = self.predict_flow5(deconv5)
        flow4 = self.predict_flow4(deconv4)
        flow3 = self.predict_flow3(deconv3)
        flow2 = self.predict_flow2(deconv2)
        
        return flow2  # 返回最終的光流預測

# 圖像的轉換操作（例如將圖像轉換為張量並調整大小）
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 調整大小
    transforms.ToTensor(),          # 轉換為張量
])

# 加載 Training 資料集
training_data = CustomDataset(root_dir='C:\\Python Project\\FlowNet\\data_scene_flow\\training\\disp_noc_0', transform=transform)
train_loader = DataLoader(training_data, batch_size=8, shuffle=True)

# 加載 Test 資料集
test_data = CustomDataset(root_dir='C:\\Python Project\\FlowNet\\data_scene_flow\\testing\\image_2', transform=transform)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# 假設我們已有模型和損失函數
model = FlowNetSimple()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 訓練迴圈
for epoch in range(20):
    model.train()
    running_loss = 0.0
    
    for images in train_loader:
        optimizer.zero_grad()  # 重置梯度
        
        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, images)  # 這裡可以替換成光流的 ground truth
        
        # 反向傳播與優化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')




