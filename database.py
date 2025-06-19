# database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

# 資料庫檔案路徑
SQLALCHEMY_DATABASE_URL = "sqlite:///./analysis_history.db"

# 建立 SQLAlchemy 引擎
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False} # SQLite 專用，允許多個執行緒同時操作
)

# 建立一個基礎類別，模型將繼承它
Base = declarative_base()

# 定義歷史記錄模型
class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True, nullable=False)
    upload_time = Column(DateTime, default=datetime.now)
    analysis_status = Column(String, nullable=False)  # 例如: "完成", "失敗", "處理中"
    final_prediction = Column(String)  # 好壞球預測結果

    # 儲存所有運動特徵 (JSON 格式)
    # 使用 Text 類型並手動處理 JSON 序列化/反序列化，因為 SQLite 的 JSON 支援有限
    all_metrics = Column(Text)

    # 影片元數據
    video_fps = Column(Float)
    video_width = Column(Integer)
    video_height = Column(Integer)

    # 其他你可能想到的東西
    # 例如：分析時長、使用者ID等
    analysis_duration_seconds = Column(Float)
    user_id = Column(String) # 如果有用戶系統

    def __repr__(self):
        return f"<AnalysisRecord(id={self.id}, filename='{self.filename}', status='{self.analysis_status}')>"

# 建立資料庫會話類別
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 建立所有定義的模型表格
def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database initialized and tables created.")

# 依存注入：獲取資料庫會話
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()