# main.py (部分修改)
# 匯入必要的套件
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
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
import warnings
from datetime import datetime # 引入 datetime
from sqlalchemy.orm import Session # 引入 Session

# 從 database.py 引入資料庫相關模組
from database import init_db, get_db, AnalysisRecord, engine, Base # 引入 engine 和 Base

warnings.filterwarnings('ignore')

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 確保在應用啟動時初始化資料庫
@app.on_event("startup")
async def startup_event():
    init_db() # 建立資料庫表

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[int, WebSocket] = {} # Use record_id to manage connections

    async def connect(self, websocket: WebSocket, record_id: int):
        await websocket.accept()
        self.active_connections[record_id] = websocket
        print(f"WebSocket connected for record_id: {record_id}")

    def disconnect(self, record_id: int):
        if record_id in self.active_connections:
            print(f"WebSocket disconnected for record_id: {record_id}")
            del self.active_connections[record_id]
        else:
            print(f"WebSocket for record_id {record_id} not found in active connections.")

    async def send_personal_message(self, message: dict, record_id: int):
        websocket = self.active_connections.get(record_id)
        if websocket:
            try:
                await websocket.send_json(message)
            except RuntimeError as e:
                print(f"Error sending message to {record_id}: {e}")
                self.disconnect(record_id) # Clean up if connection is bad

    async def broadcast(self, message: dict):
        for record_id, connection in list(self.active_connections.items()): # Iterate over a copy
            try:
                await connection.send_json(message)
            except RuntimeError as e:
                print(f"Error broadcasting to {record_id}: {e}")
                self.disconnect(record_id)


manager = ConnectionManager()

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    unique_filename = f"{os.path.splitext(file.filename)[0]}_{np.random.randint(0, 100000)}{os.path.splitext(file.filename)[1]}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    # 在檔案上傳成功後，先創建一個「處理中」的歷史記錄
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        new_record = AnalysisRecord(
            filename=file.filename, # 原始檔名
            upload_time=datetime.now(),
            analysis_status="處理中",
            final_prediction="等待分析...",
            all_metrics=json.dumps({}), # 初始為空 JSON
            video_fps=0, video_width=0, video_height=0,
            analysis_duration_seconds=0,
            user_id="anonymous" # 或從認證系統獲取
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record) # 獲取新記錄的 ID

        return JSONResponse(
            status_code=200,
            content={"message": "Video uploaded successfully", "filename": unique_filename, "record_id": new_record.id}
        )
    except Exception as e:
        print(f"Error uploading video: {e}")
        # 如果上傳失敗，可以考慮記錄為 "上傳失敗"
        failed_record = AnalysisRecord(
            filename=file.filename,
            upload_time=datetime.now(),
            analysis_status="上傳失敗",
            final_prediction="N/A",
            all_metrics=json.dumps({}),
            video_fps=0, video_width=0, video_height=0,
            analysis_duration_seconds=0,
            user_id="anonymous"
        )
        db.add(failed_record)
        db.commit()
        db.refresh(failed_record)
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {e}")


@app.websocket("/ws/analyze_video/{filename}/{record_id}")
async def analyze_video_websocket(websocket: WebSocket, filename: str, record_id: int, db: Session = Depends(get_db)):
    await manager.connect(websocket, record_id)
    video_path = os.path.join(UPLOAD_DIR, filename)

    cap = None
    pose_instance = None
    start_time = datetime.now() # 記錄分析開始時間

    # 嘗試從資料庫載入記錄 (如果已經有，更新它)
    current_record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not current_record:
        # 如果 record_id 不存在，可能是在測試，或上傳時沒有正確獲取 ID，創建一個新的
        current_record = AnalysisRecord(
            filename=filename,
            upload_time=datetime.now(),
            analysis_status="處理中",
            final_prediction="等待分析...",
            all_metrics=json.dumps({}),
            video_fps=0, video_width=0, video_height=0,
            analysis_duration_seconds=0,
            user_id="anonymous"
        )
        db.add(current_record)
        db.commit()
        db.refresh(current_record)

    try:
        if not os.path.exists(video_path):
            await manager.send_personal_message({"error": "Video file not found."}, record_id)
            current_record.analysis_status = "失敗: 檔案未找到"
            db.add(current_record)
            db.commit()
            manager.disconnect(record_id)
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            await manager.send_personal_message({"error": "Could not open video file."}, record_id)
            current_record.analysis_status = "失敗: 無法開啟影片"
            db.add(current_record)
            db.commit()
            manager.disconnect(record_id)
            return

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


        # 更新資料庫中的影片元數據
        current_record.video_fps = original_fps
        current_record.video_width = frame_width
        current_record.video_height = frame_height
        db.add(current_record)
        db.commit()

        await manager.send_personal_message(
            {"video_meta": {"fps": original_fps, "width": frame_width, "height": frame_height, "total_frames": total_frames}},
            record_id
        )

        pose_instance = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        frame_count = 0
        all_frames_metrics = [] # 用於收集每幀的指標數據
        all_frames_encoded_images = [] # 用於收集每幀的圖片數據

        # Process all frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose_instance.process(image)
            image.flags.writeable = True

            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            metrics = {}
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

                metrics = calculate_pitcher_metrics(results.pose_landmarks.landmark)
                for key, value in metrics.items():
                    if isinstance(value, np.integer):
                        metrics[key] = int(value)
                    elif isinstance(value, np.floating):
                        metrics[key] = float(value)

            # Collect metrics and image data for each frame
            all_frames_metrics.append({"frame_num": frame_count, "metrics": metrics})

            _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            all_frames_encoded_images.append(encoded_image)

            # Send progress update
            if frame_count % 10 == 0 or frame_count == total_frames: # Update every 10 frames or at the end
                progress = int((frame_count / total_frames) * 100)
                await manager.send_personal_message({"progress": progress, "current_frame_num": frame_count}, record_id)


        # After all frames are processed
        if all_frames_metrics:
            final_prediction_result = predict_video(video_path)

        # Update database record
        current_record.analysis_status = "完成"
        current_record.final_prediction = str(final_prediction_result)
        current_record.all_metrics = json.dumps(all_frames_metrics) # 儲存所有幀的指標數據
        end_time = datetime.now()
        current_record.analysis_duration_seconds = (end_time - start_time).total_seconds()
        db.add(current_record)
        db.commit()
        print(f"Record for {current_record.filename} updated to status: {current_record.analysis_status}, ID: {current_record.id}")

        # Send all collected data at once
        await manager.send_personal_message({
            "final_predict": str(final_prediction_result),
            "all_frames_data": {
                "metrics": all_frames_metrics,
                "images": all_frames_encoded_images,
                "video_meta": {"fps": original_fps, "width": frame_width, "height": frame_height, "total_frames": total_frames}
            }
        }, record_id)

    except WebSocketDisconnect:
        print(f"WebSocket client disconnected during analysis.")
        current_record.analysis_status = "中斷連線"
        db.add(current_record)
        db.commit()
    except Exception as e:
        print(f"Error during video analysis: {e}")
        import traceback
        traceback.print_exc()
        await manager.send_personal_message({"error": f"Server error during analysis: {e}"}, record_id)
        current_record.analysis_status = "失敗: " + str(e)[:200] # 儲存部分錯誤訊息
        db.add(current_record)
        db.commit()
    finally:
        if cap is not None:
            cap.release()
        if pose_instance is not None:
            pose_instance.close()

        # 這裡根據需求決定是否刪除影片
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Video file {video_path} deleted successfully.")
            except Exception as e:
                print(f"Error deleting video file {video_path}: {e}")
        manager.disconnect(record_id)
        print(f"Analysis for {filename} finished.")


# 新增 API 端點，用於獲取歷史記錄 (簡要列表)
@app.get("/api/history")
async def get_history_records(db: Session = Depends(get_db)):
    # 獲取最近的 5 條記錄，按時間倒序排列
    records = db.query(AnalysisRecord).order_by(AnalysisRecord.upload_time.desc()).limit(5).all()
    
    # 將資料庫物件轉換為可序列化的字典列表
    history_data = []
    for record in records:
        history_data.append({
            "id": record.id,
            "filename": record.filename,
            "upload_time": record.upload_time.strftime("%H:%M:%S"), # 格式化時間
            "analysis_status": record.analysis_status,
            "final_prediction": record.final_prediction,
            "analysis_duration_seconds": record.analysis_duration_seconds
            # 注意：all_metrics 由於可能很大，通常不一次性傳給前端，
            # 如果需要，可以為單一記錄提供一個詳細 API
        })
    return JSONResponse(content=history_data)

# 新增 API 端點，用於獲取單一歷史記錄的詳細資訊 (包括 all_metrics)
@app.get("/api/history/{record_id}")
async def get_history_record_by_id(record_id: int, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not record:
        print('not record')
        raise HTTPException(status_code=404, detail="Record not found")

    # 從 JSON 字串解析 all_metrics 為 Python 物件
    all_metrics_data = json.loads(record.all_metrics) if record.all_metrics else []
    print('have record')
    return JSONResponse(content={
        "id": record.id,
        "filename": record.filename,
        "upload_time": record.upload_time.strftime("%Y-%m-%d %H:%M:%S"), # 更詳細的時間格式
        "analysis_status": record.analysis_status,
        "final_prediction": record.final_prediction,
        "analysis_duration_seconds": record.analysis_duration_seconds,
        "video_fps": record.video_fps,
        "video_width": record.video_width,
        "video_height": record.video_height,
        "all_metrics": all_metrics_data # 包含已解析的指標數據
    })

# ... (運動力學指標的計算函式區塊保持不變) ...
def get_landmark_vector(landmarks_list, idx):
    return np.array([landmarks_list[idx].x, landmarks_list[idx].y, landmarks_list[idx].z])

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

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

def calculate_pitcher_metrics(landmarks_mp_obj: list) -> dict:
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)