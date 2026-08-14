import json

from cadebot.rag.db_source import chunk_database, get_menu_data


def test_get_menu_data_returns_three_sections():
    data = get_menu_data()
    assert set(data.keys()) == {"menu", "promotions", "faqs"}
    assert len(data["menu"]) == 12
    assert len(data["promotions"]) == 3
    assert len(data["faqs"]) == 20


def test_menu_chunk_ids_use_item_code():
    chunks = chunk_database()
    latte = next(c for c in chunks if c.id == "menu:VR_LATTE_M")
    assert "55,000" in latte.text or "55.000" in latte.text
    assert "Viva Latte" in latte.text


def test_promo_and_faq_chunks_have_correct_id_prefix():
    chunks = chunk_database()
    assert any(c.id.startswith("promo:") for c in chunks)
    assert any(c.id.startswith("faq:db_") for c in chunks)


def test_attributes_json_is_flattened_into_readable_text():
    chunks = chunk_database()
    latte = next(c for c in chunks if c.id == "menu:VR_LATTE_M")
    # câu hỏi "có size L không" chỉ match được nếu size đã được trải phẳng
    assert "Size" in latte.text
    assert "L" in latte.text


def test_db_chunk_ids_do_not_collide_with_markdown_chunk_ids():
    from cadebot.rag.chunker import chunk_all_markdown

    all_ids = [c.id for c in chunk_database()] + [c.id for c in chunk_all_markdown()]
    assert len(all_ids) == len(set(all_ids))


def test_unavailable_items_are_excluded():
    # KB không được quảng cáo món đã ngừng bán
    chunks = chunk_database()
    data = get_menu_data()
    assert len([c for c in chunks if c.id.startswith("menu:")]) == len(data["menu"])
