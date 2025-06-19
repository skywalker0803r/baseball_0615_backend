import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import numpy as np
import random
from PIL import Image
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

def predict_video(video_path=None, model=None, transform=None, original_frames=240, sample_frames=120):
    import random
    def 隨機回傳好壞球():
        return random.choice(['好球', '壞球'])
    return 隨機回傳好壞球()

if __name__ == '__main__':
    print(predict_video())