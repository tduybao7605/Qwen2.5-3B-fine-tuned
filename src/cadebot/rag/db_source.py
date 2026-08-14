"""Đọc menu/khuyến mãi/FAQ từ database và sinh chunk.

GIAI ĐOẠN 7 (DB thật của quán): CHỈ sửa bên trong get_menu_data().
Toàn bộ logic chunk + sync phía dưới giữ nguyên.
"""
import json
import sqlite3

from cadebot.rag import config
from cadebot.rag.chunker import Chunk


def _available_column(conn: sqlite3.Connection) -> str:
    """schema.sql (Postgres) dùng `is_available`, demo_cafe.db (SQLite) dùng `available`."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(menu_items)")}
    for candidate in ("available", "is_available"):
        if candidate in cols:
            return candidate
    raise RuntimeError(f"menu_items không có cột available/is_available. Có: {sorted(cols)}")


def get_menu_data() -> dict:
    conn = sqlite3.connect(config.DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        avail = _available_column(conn)
        return {
            "menu": conn.execute(
                f"SELECT * FROM menu_items WHERE {avail} = 1 ORDER BY id"
            ).fetchall(),
            "promotions": conn.execute("SELECT * FROM promotions ORDER BY id").fetchall(),
            "faqs": conn.execute("SELECT * FROM faqs ORDER BY id").fetchall(),
        }
    finally:
        conn.close()


_ATTR_LABELS = {
    "sizeOptions": "Size",
    "sweetnessOptions": "Độ ngọt",
    "iceOptions": "Tùy chọn đá",
    "temperatureOptions": "Nóng/Lạnh",
    "toppings": "Topping",
    "toppingOptions": "Topping",
}


def _flatten_attributes(raw: str | None) -> str:
    """JSON attributes -> dòng tiếng Việt để embedding bắt được."""
    if not raw:
        return ""
    try:
        attrs = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""

    lines = []
    if attrs.get("caffeine") is True:
        lines.append("  + Có caffeine")
    elif attrs.get("caffeine") is False:
        lines.append("  + Không có caffeine")

    for key, label in _ATTR_LABELS.items():
        values = attrs.get(key)
        if isinstance(values, list) and values:
            lines.append(f"  + {label}: {', '.join(str(v) for v in values)}")
    return "\n".join(lines)


def chunk_database() -> list[Chunk]:
    data = get_menu_data()
    chunks: list[Chunk] = []

    for row in data["menu"]:
        body = (
            f"Món: {row['name']} (Mã: {row['item_code']} | Nhóm: {row['category']})\n"
            f"  + Giá: {row['price']:,} VNĐ\n"
            f"  + Mô tả: {row['description']}\n"
            f"  + Phương pháp: {row['brewing_method']}"
        )
        attrs = _flatten_attributes(row["attributes"] if "attributes" in row.keys() else None)
        if attrs:
            body += "\n" + attrs
        chunks.append(Chunk(id=f"menu:{row['item_code']}", text=body, source="demo_cafe.db"))

    for row in data["promotions"]:
        body = (
            f"Khuyến mãi: {row['title']} (Mã: {row['promo_code']})\n"
            f"  + Chi tiết: {row['discount_detail']}\n"
            f"  + Thời gian: {row['start_date']} đến {row['end_date']}\n"
            f"  + Điều kiện: {row['conditions']}"
        )
        chunks.append(Chunk(id=f"promo:{row['promo_code']}", text=body, source="demo_cafe.db"))

    for row in data["faqs"]:
        body = f"Q: {row['question']}\nA: {row['answer']}"
        # tiền tố db_ để không đụng id faq:md_* của chunker markdown
        chunks.append(Chunk(id=f"faq:db_{row['faq_id']}", text=body, source="demo_cafe.db"))

    return chunks
