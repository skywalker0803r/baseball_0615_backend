# 匯入必要的套件
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException,BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import mediapipe as mp
import numpy as np
import os
import asyncio
import json
from dotenv import load_dotenv
import base64
from predict import predict_video

# 載入 .env 檔案中的環境變數（例如設定密鑰或其他配置）
load_dotenv()

# 建立 FastAPI 應用
app = FastAPI()

# 開啟跨來源請求（CORS），允許所有來源與請求方法，因為前端部署與後端網域不同
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可替換為前端實際的 URL 以增加安全性
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 MediaPipe 的 Pose 模組和繪圖工具
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 設定影片上傳後的儲存資料夾
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 管理 WebSocket 連線的類別
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    # 新增連線
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected: {websocket.client}")

    # 移除連線
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"WebSocket disconnected: {websocket.client}")

    # 傳送訊息給單一客戶端
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    # 廣播訊息給所有連線中的用戶
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# 影片上傳 API：接收前端傳來的影片並儲存至伺服器
@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...),background_tasks: BackgroundTasks = BackgroundTasks()):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    return JSONResponse(status_code=200, content={"message": "Video uploaded successfully", "filename": file.filename,"predict":str(predict_video(file_path))})
    
# WebSocket 分析端點：串流分析影片內容並回傳畫面 + 姿勢資訊 + 運動指標
@app.websocket("/ws/analyze_video/{filename}")
async def analyze_video_websocket(websocket: WebSocket, filename: str):
    await manager.connect(websocket)
    video_path = os.path.join(UPLOAD_DIR, filename)

    cap = None  # 用於讀取影片幀的變數
    pose_instance = None  # MediaPipe Pose 實例

    try:
        # 確認影片是否存在
        if not os.path.exists(video_path):
            await manager.send_personal_message({"error": "Video file not found."}, websocket)
            manager.disconnect(websocket)
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            await manager.send_personal_message({"error": "Could not open video file."}, websocket)
            manager.disconnect(websocket)
            return

        # 初始化 MediaPipe Pose 偵測器
        pose_instance = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        frame_count = 0  # 計算幀數
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 沒有更多幀了

            frame_count += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV 預設為 BGR，需要轉為 RGB
            image.flags.writeable = False  # 設為唯讀，提高效率
            results = pose_instance.process(image)  # 進行姿勢偵測
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            landmarks_data = []
            if results.pose_landmarks:
                # 繪製姿勢骨架
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks_data.append(lm)

                # 計算投球動作指標
                metrics = calculate_pitcher_metrics(landmarks_data)
                for key, value in metrics.items():
                    if isinstance(value, np.integer):
                        metrics[key] = int(value)
                    elif isinstance(value, np.floating):
                        metrics[key] = float(value)

            # 將每一幀轉為 base64 圖片串流回前端
            _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            await manager.send_personal_message(
                {
                    "frame_data": jpg_as_text,
                    "frame_num": frame_count,
                    "landmarks": [
                        {
                            "id": id,
                            "x": lm.x, "y": lm.y, "z": lm.z,
                            "visibility": lm.visibility,
                            "px": int(lm.x * frame.shape[1]),
                            "py": int(lm.y * frame.shape[0])
                        } for id, lm in enumerate(landmarks_data)
                    ],
                    "metrics": metrics if 'metrics' in locals() else {}
                },
                websocket
            )
            await asyncio.sleep(0.01)  # 控制串流速度，避免前端過載

    except WebSocketDisconnect:
        print(f"WebSocket client disconnected during analysis.")
    except Exception as e:
        print(f"Error during video analysis: {e}")
        await manager.send_personal_message({"error": f"Server error during analysis: {e}"}, websocket)
    finally:
        if cap is not None:
            cap.release()
        if pose_instance is not None:
            pose_instance.close()
        if os.path.exists(video_path):
            os.remove(video_path)  # 清除上傳的影片檔案，避免佔空間
        manager.disconnect(websocket)
        print(f"Analysis for {filename} finished.")

# --- 以下為運動力學指標的計算函式區塊 ---

# 將 landmark 轉為 numpy 向量
def get_landmark_vector(landmark, idx):
    return np.array([landmark[idx].x, landmark[idx].y, landmark[idx].z])

# 計算三點夾角 (例如：肩膀-手肘-手腕)
def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# 各種身體部位或姿勢角度的運動指標函式
def calc_stride_angle(lm): return calculate_angle(get_landmark_vector(lm, 24), get_landmark_vector(lm, 26), get_landmark_vector(lm, 23))
def calc_throwing_angle(lm): return calculate_angle(get_landmark_vector(lm, 12), get_landmark_vector(lm, 14), get_landmark_vector(lm, 16))
def calc_arm_symmetry(lm): return 1 - abs(lm[15].y - lm[16].y)
def calc_hip_rotation(lm): return abs(lm[23].z - lm[24].z)
def calc_elbow_height(lm): return lm[14].y
def calc_ankle_height(lm): return lm[28].y
def calc_shoulder_rotation(lm): return abs(lm[11].z - lm[12].z)
def calc_torso_tilt_angle(lm): return calculate_angle(get_landmark_vector(lm, 11), get_landmark_vector(lm, 23), get_landmark_vector(lm, 24))
def calc_release_distance(lm): return np.linalg.norm(get_landmark_vector(lm, 16) - get_landmark_vector(lm, 12))
def calc_shoulder_to_hip(lm): return abs(lm[12].x - lm[24].x)

# 綜合運算所有指標，回傳 dict 結果
def calculate_pitcher_metrics(landmarks_data: list) -> dict:
    metric_funcs = {
        "stride_angle":       calc_stride_angle,
        "throwing_angle":     calc_throwing_angle,
        "arm_symmetry":       calc_arm_symmetry,
        "hip_rotation":       calc_hip_rotation,
        "elbow_height":       calc_elbow_height,
        "ankle_height":       calc_ankle_height,
        "shoulder_rotation":  calc_shoulder_rotation,
        "torso_tilt_angle":   calc_torso_tilt_angle,
        "release_distance":   calc_release_distance,
        "shoulder_to_hip":    calc_shoulder_to_hip,
    }

    return {
        name: float(round(func(landmarks_data), 2))
        for name, func in metric_funcs.items()
    }

# 開發測試用：執行 FastAPI 應用（使用 `python this_file.py` 時啟動）
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
