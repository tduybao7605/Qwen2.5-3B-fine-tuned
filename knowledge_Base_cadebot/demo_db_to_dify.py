import sqlite3
import requests
import json
import os

DB_FILE = "demo_cafe.db"

def get_menu_data_from_db():
    """
    Hàm đọc dữ liệu thực tế từ Database (menu_items, promotions, faqs)
    và chuyển đổi thành định dạng Markdown chuẩn cho Dify Knowledge Base.
    """
    if not os.path.exists(DB_FILE):
        print(f"Không tìm thấy file Database: {DB_FILE}")
        return ""

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 1. Đọc thực đơn
    cursor.execute("SELECT item_code, name, category, price, description, brewing_method FROM menu_items WHERE available = 1")
    items = cursor.fetchall()
    
    # 2. Đọc khuyến mãi
    cursor.execute("SELECT promo_code, title, discount_detail, start_date, end_date, conditions FROM promotions")
    promos = cursor.fetchall()

    # 3. Đọc FAQ
    cursor.execute("SELECT question, answer FROM faqs")
    faqs = cursor.fetchall()

    conn.close()

    # Chuyển đổi dữ liệu sang văn bản Markdown
    md_content = "# DANH MỤC THỰC ĐƠN, KHUYẾN MÃI VÀ FAQ VIVA RESERVE (ĐỒNG BỘ TỪ DATABASE)\n\n"
    
    md_content += "## 1. Danh sách thực đơn đồ uống & món ăn\n"
    for item in items:
        code, name, category, price, desc, method = item
        md_content += f"- **{name}** (Mã: {code} | Nhóm: {category})\n"
        md_content += f"  + Giá: {price:,} VNĐ\n"
        md_content += f"  + Mô tả: {desc}\n"
        md_content += f"  + Phương pháp: {method}\n\n"

    md_content += "## 2. Chương trình khuyến mãi & Combo ưu đãi\n"
    for promo in promos:
        code, title, detail, start, end, cond = promo
        md_content += f"- **{title}** (Mã: {code})\n"
        md_content += f"  + Chi tiết: {detail}\n"
        md_content += f"  + Thời gian: Từ {start} đến {end}\n"
        md_content += f"  + Điều kiện: {cond}\n\n"

    md_content += "## 3. Câu hỏi thường gặp (FAQ)\n"
    for faq in faqs:
        q, a = faq
        md_content += f"Q: {q}\n"
        md_content += f"A: {a}\n\n"

    return md_content

def sync_to_dify_dataset(dify_api_key: str, dataset_id: str, dify_base_url: str = "http://localhost/v1"):
    """
    Hàm gọi Dify Dataset API để đẩy dữ liệu tự động vào Knowledge Base trên Dify
    """
    text_data = get_menu_data_from_db()
    if not text_data:
        print("❌ Không có dữ liệu để đồng bộ!")
        return

    url = f"{dify_base_url.rstrip('/')}/datasets/{dataset_id}/document/create_by_text"
    headers = {
        "Authorization": f"Bearer {dify_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "name": "Live_Database_Menu_And_Promotions.md",
        "text": text_data,
        "indexing_technique": "high_quality",
        "process_rule": {
            "mode": "automatic"
        }
    }
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        if res.status_code == 200:
            print("Da dong bo du lieu thuc don tu DB sang Dify Knowledge Base thanh cong!")
            print(res.json())
        else:
            print(f"Loi khi goi Dify Dataset API ({res.status_code}): {res.text}")
    except Exception as e:
        print(f"Loi ket noi API: {e}")

if __name__ == "__main__":
    # Đọc DB và xuất ra file Markdown để kiểm tra
    output_text = get_menu_data_from_db()
    with open("db_exported_kb.md", "w", encoding="utf-8") as f:
        f.write(output_text)
    print("Da xuat du lieu tu Database ra file 'db_exported_kb.md' thanh cong!")
