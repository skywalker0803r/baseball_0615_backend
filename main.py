# 匯入必要的套件
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse # 導入 FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import mediapipe as mp
import numpy as np
import os
import asyncio
import json
from dotenv import load_dotenv
import base64 # 重新啟用 base64
from predict import predict_video # 引入預測模型

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
mp_drawing = mp.solutions.drawing_utils # 重新啟用繪圖工具

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
        # 尋找並移除正確的連線物件
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"WebSocket disconnected: {websocket.client}")
        else:
            print(f"WebSocket {websocket.client} not found in active connections.")


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
async def upload_video(file: UploadFile = File(...)):
    # 確保影片檔名是唯一的，避免衝突
    unique_filename = f"{os.path.splitext(file.filename)[0]}_{np.random.randint(0, 100000)}{os.path.splitext(file.filename)[1]}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return JSONResponse(
        status_code=200,
        content={"message": "Video uploaded successfully", "filename": unique_filename}
    )

# WebSocket 分析端點：串流分析影片內容並回傳姿勢資訊 + 運動指標
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

        # 獲取影片的原始 FPS 和尺寸
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 傳送影片元數據給前端，前端將用此設置 Canvas 尺寸
        await manager.send_personal_message(
            {"video_meta": {"fps": original_fps, "width": frame_width, "height": frame_height}},
            websocket
        )

        # 初始化 MediaPipe Pose 偵測器
        pose_instance = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        frame_count = 0  # 計算幀數
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 沒有更多幀了

            frame_count += 1
            
            # 將 OpenCV BGR 影像轉為 RGB，供 MediaPipe 處理
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # 設為唯讀，提高效率
            results = pose_instance.process(image)  # 進行姿勢偵測
            image.flags.writeable = True # 處理完後可寫回
            
            # 將圖像轉回 BGR 以便 OpenCV 繪製 (或直接在 RGB 上繪製，然後轉為 BGR 再編碼)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            landmarks_data = []
            metrics = {} # Initialize metrics to an empty dict
            
            # 如果偵測到骨架點
            if results.pose_landmarks:
                # 在圖像上繪製骨架
                mp_drawing.draw_landmarks(
                    image_bgr, # 在 BGR 圖像上繪製
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), # 點的顏色 (BGR: 藍色)
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # 線的顏色 (BGR: 綠色)
                )

                for id, lm in enumerate(results.pose_landmarks.landmark):
                    # 注意：lm.x 和 lm.y 是正規化座標 (0~1)
                    landmarks_data.append(
                        {
                            "id": id,
                            "x": lm.x, "y": lm.y, "z": lm.z,
                            "visibility": lm.visibility
                        }
                    )

                # 計算投球動作指標 (傳遞原始的 MediaPipe landmark 物件列表)
                metrics = calculate_pitcher_metrics(results.pose_landmarks.landmark)
                # 確保 metrics 中的 numpy 數值類型被轉換為 Python 原生類型，以便 JSON 序列化
                for key, value in metrics.items():
                    if isinstance(value, np.integer):
                        metrics[key] = int(value)
                    elif isinstance(value, np.floating):
                        metrics[key] = float(value)

            # 將處理後的圖像 (帶骨架) 編碼為 Base64
            # 將圖像質量設置為較低以減少數據量，權衡視覺效果和傳輸速度
            _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70]) # 質量調整為 70
            encoded_image = base64.b64encode(buffer).decode('utf-8')

            # 傳送包含 Base64 圖像、幀號和指標數據的 JSON
            await manager.send_personal_message(
                {
                    "frame_num": frame_count,
                    "image_data": encoded_image, # 傳送 Base64 編碼的圖像
                    "metrics": metrics
                },
                websocket
            )
            # 控制骨架數據發送速度，盡量以原始 FPS 速度發送
            # 這一步對於"影片不要慢動作"非常關鍵
            # 如果後端處理速度跟不上 original_fps，這裡仍然會是瓶頸
            # await asyncio.sleep(1 / original_fps)

        # 🔽 結束時送出分析結果
        # predict_video 是一個同步函式，會阻塞事件迴圈，但因為在影片結束後才調用，影響較小
        final_result = predict_video(video_path)
        await manager.send_personal_message({"final_predict": str(final_result)}, websocket)

    except WebSocketDisconnect:
        print(f"WebSocket client disconnected during analysis.")
    except Exception as e:
        print(f"Error during video analysis: {e}")
        import traceback
        traceback.print_exc() # 印出完整的錯誤追蹤，方便除錯
        await manager.send_personal_message({"error": f"Server error during analysis: {e}"}, websocket)
    finally:
        if cap is not None:
            cap.release()
        if pose_instance is not None:
            pose_instance.close()
        # **重要：這裡不刪除影片檔案，因為可能需要再次分析**
        # if os.path.exists(video_path):
        #     os.remove(video_path)
        manager.disconnect(websocket)
        print(f"Analysis for {filename} finished.")

# --- 以下為運動力學指標的計算函式區塊 ---
# (這部分與您原先的 predict.py 和 main.py 中的邏輯相同，用於計算運動指標)

# 將 landmark 轉為 numpy 向量
# landmark 是一個 MediaPipe landmark object，這裡的 lm 是單一 landmark 物件
def get_landmark_vector(landmarks_list, idx):
    # landmarks_list 是 results.pose_landmarks.landmark (MediaPipe 的列表結構)
    return np.array([landmarks_list[idx].x, landmarks_list[idx].y, landmarks_list[idx].z])

# 計算三點夾角 (例如：肩膀-手肘-手腕)
def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# 各種身體部位或姿勢角度的運動指標函式
# 這些函式現在接收 `landmarks_mp_obj` (即 `results.pose_landmarks.landmark`)
def calc_stride_angle(lm): return calculate_angle(get_landmark_vector(lm, 24), get_landmark_vector(lm, 26), get_landmark_vector(lm, 23))
def calc_throwing_angle(lm): return calculate_angle(get_landmark_vector(lm, 12), get_landmark_vector(lm, 14), get_landmark_vector(lm, 16))
def calc_arm_symmetry(lm): return 1 - abs(lm[15].y - lm[16].y) # 調整為直接使用 y 座標
def calc_hip_rotation(lm): return abs(lm[23].z - lm[24].z) # 調整為直接使用 z 座標
def calc_elbow_height(lm): return lm[14].y # 調整為直接使用 y 座標
def calc_ankle_height(lm): return lm[28].y # 調整為直接使用 y 座標
def calc_shoulder_rotation(lm): return abs(lm[11].z - lm[12].z) # 調整為直接使用 z 座標
def calc_torso_tilt_angle(lm): return calculate_angle(get_landmark_vector(lm, 11), get_landmark_vector(lm, 23), get_landmark_vector(lm, 24))
def calc_release_distance(lm): return np.linalg.norm(get_landmark_vector(lm, 16) - get_landmark_vector(lm, 12)) # 調整為直接使用向量計算
def calc_shoulder_to_hip(lm): return abs(lm[12].x - lm[24].x) # 調整為直接使用 x 座標

# 綜合運算所有指標，回傳 dict 結果
def calculate_pitcher_metrics(landmarks_mp_obj: list) -> dict: # 這裡現在接收 MediaPipe landmark 物件列表
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
        name: float(round(func(landmarks_mp_obj), 2))
        for name, func in metric_funcs.items()
    }

# 開發測試用：執行 FastAPI 應用（使用 `python this_file.py` 時啟動）
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)