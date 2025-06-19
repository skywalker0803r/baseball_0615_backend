# view_db.py
import sqlite3
import json
from datetime import datetime

# 資料庫檔案路徑
DB_FILE = 'analysis_history.db'

def view_analysis_records(db_file):
    """
    連接到 SQLite 資料庫，並列印出 analysis_records 表格中的所有內容。
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # 查詢所有 analysis_records 表格中的資料
        cursor.execute("SELECT * FROM analysis_records ORDER BY upload_time DESC")
        records = cursor.fetchall()

        if not records:
            print(f"資料庫 '{db_file}' 中的 'analysis_records' 表格目前沒有資料。")
            return

        # 獲取欄位名稱
        column_names = [description[0] for description in cursor.description]
        print(f"從 '{db_file}' 的 'analysis_records' 表格中撈取到的資料：\n")

        for record in records:
            record_dict = {}
            for i, col_name in enumerate(column_names):
                value = record[i]
                if col_name == "upload_time" and value:
                    try:
                        # 假設時間是以 ISO 格式儲存
                        value = datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        pass # 如果格式不對，就保持原樣

                if col_name == "all_metrics" and value:
                    try:
                        # 嘗試解析 JSON 字符串
                        value = json.loads(value)
                        # 如果 all_metrics 包含多幀數據，這裡可以只顯示前幾幀或一些摘要
                        if isinstance(value, list) and len(value) > 0:
                            print(f"    {col_name}: (包含 {len(value)} 幀數據，部分預覽)")
                            # 只顯示前兩幀的數據，避免輸出過長
                            for j, frame_data in enumerate(value[:2]):
                                print(f"        幀 {frame_data.get('frame_num', 'N/A')}: {frame_data.get('metrics', {})}")
                            if len(value) > 2:
                                print("        ...")
                            continue # 跳過對 all_metrics 再次列印
                    except json.JSONDecodeError:
                        pass # 如果不是有效的 JSON，就保持原樣

                record_dict[col_name] = value

            # 列印每條記錄的詳細資訊
            for key, val in record_dict.items():
                print(f"    {key}: {val}")
            print("-" * 50) # 分隔線

    except sqlite3.Error as e:
        print(f"資料庫操作錯誤: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    view_analysis_records(DB_FILE)