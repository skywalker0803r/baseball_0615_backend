import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pytorchvideo.models.hub import x3d_xs
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_video
from torchvision import transforms as T
import cv2
import numpy as np
from torchvision import transforms
import random
from PIL import Image
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
import copy
from PIL import Image, ImageFilter
import json
import warnings
warnings.filterwarnings('ignore')

def read_video_cv2(path, max_frames=240, sample_frames=120):
    # 開啟影片檔案
    cap = cv2.VideoCapture(path)
    frames = []

    # 逐幀讀取影片直到結束
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 如果讀不到（影片結束），就跳出
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR轉RGB
        frames.append(frame)

    # 釋放影片資源
    cap.release()

    # 若讀不到任何幀，丟出錯誤
    total_frames = len(frames)
    if total_frames == 0:
        raise RuntimeError(f"Cannot read video {path}")

    # 若影片幀數不足 max_frames，重複最後一幀來補足長度
    while len(frames) < max_frames:
        frames.append(frames[-1].copy())

    # 限制最多只取 max_frames 幀
    frames = frames[:max_frames]

    # 等距地選出 sample_frames 幀索引位置
    indices = np.linspace(0, max_frames - 1, sample_frames).astype(int)

    # 根據索引取出對應的幀
    sampled_frames = [frames[i] for i in indices]

    # 將幀列表轉成 NumPy 陣列 (T, H, W, C)
    video_np = np.stack(sampled_frames, axis=0)

    # 轉成 PyTorch Tensor 並重新排列維度為 (C, T, H, W)
    video_t = torch.from_numpy(video_np).permute(3, 0, 1, 2)

    # 回傳影片張量
    return video_t

# 資料模型
class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()  # 繼承自 nn.Module 的初始化
        # 將 mean 轉成張量，並 reshape 成 (C, 1, 1, 1)，方便與 (C, T, H, W) 的影像進行 broadcast
        self.mean = torch.tensor(mean).view(-1, 1, 1, 1)
        # 將 std 轉成張量，並 reshape 成 (C, 1, 1, 1)，方便與影像資料進行除法 broadcast
        self.std = torch.tensor(std).view(-1, 1, 1, 1)

    def forward(self, x):
        # 對輸入張量 x 做標準化：每個 channel 減去平均值、除以標準差
        return (x - self.mean) / self.std


# 🔧 安全資料增強
class SafeVideoAugmentation:
    def __init__(self, resize=(224, 224), apply_blur_prob=0.3, apply_brightness_prob=0.3):
        # 設定每幀影像的 resize 大小
        self.resize = resize

        # 定義將 PIL 影像轉成 tensor 的轉換器 (0~1 範圍)
        self.to_tensor = transforms.ToTensor()

        # 定義應用模糊的機率
        self.apply_blur_prob = apply_blur_prob

        # 定義應用亮度調整的機率
        self.apply_brightness_prob = apply_brightness_prob

        # 定義影片的標準化方式 (使用 ImageNet 標準)
        self.normalize = Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])

    def __call__(self, frames):
        augmented = []  # 用來儲存增強後的每一幀

        # 隨機決定是否對整個影片應用模糊
        apply_blur = random.random() < self.apply_blur_prob

        # 隨機決定是否對整個影片應用亮度調整
        apply_brightness = random.random() < self.apply_brightness_prob

        # 隨機產生一個亮度調整係數 (0.8~1.2 之間)
        brightness_factor = random.uniform(0.8, 1.2)

        # 遍歷每一幀做增強處理
        for frame in frames:
            # 將影像 resize 成固定大小
            frame = cv2.resize(frame, self.resize)

            # 將 NumPy 陣列轉為 PIL 影像，方便後續使用 PIL 的影像增強方法
            pil_frame = Image.fromarray(frame)

            # 若要套用模糊，則加上 GaussianBlur
            if apply_blur:
                pil_frame = pil_frame.filter(ImageFilter.GaussianBlur(radius=1.5))  # radius 控制模糊強度

            # 若要套用亮度調整，則進行亮度增強
            if apply_brightness:
                pil_frame = transforms.functional.adjust_brightness(pil_frame, brightness_factor)

            # ✅ 將彩色影像轉為灰階（單通道）再轉回 RGB（三通道）
            gray_frame = pil_frame.convert("L")         # 轉灰階
            gray_frame = gray_frame.convert("RGB")      # 再轉回 RGB（R=G=B）

            # 將 PIL 影像轉成 tensor，形狀為 (C, H, W)，值在 0~1
            tensor_frame = self.to_tensor(gray_frame)

            # 加入增強後的幀
            augmented.append(tensor_frame)

        # 將所有幀堆疊起來成 (T, C, H, W)
        augmented_tensor = torch.stack(augmented)

        # 將維度轉換成 (C, T, H, W)，符合影片模型輸入格式
        augmented_tensor = augmented_tensor.permute(1, 0, 2, 3)

        # 對整段影片張量做標準化處理
        augmented_tensor = self.normalize(augmented_tensor)

        # 回傳增強與標準化後的影片張量
        return augmented_tensor #我希望將影片轉灰階但是保留通道數

# 這裡將模型model_path的權重載入
model_path = './model/epoch_2_valacc_0.5646.pt'
model = x3d_xs(pretrained=True)  # 載入預訓練的X3D XS模型
model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, 2)  # 修改最後分類層為2分類
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判斷是否使用GPU，否則用CPU
model = model.to(device)  # 將模型搬移至指定裝置（GPU或CPU）

# 載入已有模型權重
model.load_state_dict(torch.load(model_path, map_location=device))
print(f"Model {model_path} weights loaded successfully.")
def predict_video(video_path, model=model, transform=None, original_frames=240, sample_frames=120):
    # 預設使用 SafeVideoAugmentation，關閉 blur/亮度調整（驗證時也這樣）
    if transform is None:
        transform = SafeVideoAugmentation(apply_blur_prob=0.0, apply_brightness_prob=0.0)
    
    # 讀影片 → (C, T, H, W)
    video = read_video_cv2(video_path, original_frames, sample_frames)

    # (C, T, H, W) → (T, H, W, C) → numpy 給 transform 用
    video = video.permute(1, 2, 3, 0).numpy()
    
    # 資料增強（其實只有 resize + 灰階 + normalize）
    video = transform(video)  # 回傳 (C, T, H, W)

    # 加 batch 維度 (1, C, T, H, W)
    video = video.unsqueeze(0).to(device)

    # 推論
    with torch.no_grad():
        outputs = model(video)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_label = np.argmax(probs)
        if pred_label == 0:
            pred_label = '壞球'
        if pred_label == 1:
            pred_label = '好球'
    return pred_label