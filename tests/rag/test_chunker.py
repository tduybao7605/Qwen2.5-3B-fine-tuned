from rag import config
from rag.chunker import Chunk, chunk_all_markdown, chunk_faq_file, chunk_markdown_file


def test_chunk_render_prefixes_id():
    c = Chunk(id="menu:VR_LATTE", text="Giá 55.000đ", source="test")
    assert c.render() == "[menu:VR_LATTE]\nGiá 55.000đ"


def test_faq_file_yields_one_chunk_per_qa_pair():
    chunks = chunk_faq_file(config.KB_DIR / "05_Bo_Cau_Hoi_Thuong_Gap_FAQ.md")
    # Thực tế: `grep -c '^Q:'` trên file 05 trả về 20 (plan ghi nhầm 21).
    assert len(chunks) == 20
    assert all(c.id.startswith("faq:md_") for c in chunks)
    # mỗi chunk phải chứa CẢ câu hỏi lẫn câu trả lời
    assert all("Q:" in c.text and "A:" in c.text for c in chunks)


def test_faq_chunk_keeps_question_and_answer_together():
    chunks = chunk_faq_file(config.KB_DIR / "05_Bo_Cau_Hoi_Thuong_Gap_FAQ.md")
    latte = next(c for c in chunks if "Viva Latte có vị như thế nào" in c.text)
    assert "sữa béo nhẹ" in latte.text


def test_section_file_splits_on_h2():
    chunks = chunk_markdown_file(config.KB_DIR / "01_Tong_Quan_Thuong_Hieu.md")
    # file 01 có 3 section `##`
    assert len(chunks) == 3
    assert chunks[0].id == "doc:01_Tong_Quan_Thuong_Hieu#1"


def test_every_chunk_carries_document_title_for_context():
    chunks = chunk_markdown_file(config.KB_DIR / "01_Tong_Quan_Thuong_Hieu.md")
    # chunk lẻ mất ngữ cảnh nếu không mang tiêu đề file
    assert all("VIVA RESERVE" in c.text.upper() for c in chunks)


def test_menu_file_splits_on_h3_subgroups():
    chunks = chunk_markdown_file(config.KB_DIR / "02_Menu_Va_Phuong_Phap_Pha_Che.md")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk id phải duy nhất"
    joined = " ".join(c.text for c in chunks)
    assert "Trà Đào Cam Sả" in joined
    assert "Extra Shot" in joined  # section toppings không được rơi mất


def test_chunk_all_markdown_ids_are_globally_unique():
    chunks = chunk_all_markdown()
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert len(chunks) > 25


def test_no_chunk_is_empty_or_whitespace_only():
    for c in chunk_all_markdown():
        assert c.text.strip(), f"chunk rỗng: {c.id}"


def test_no_chunk_exceeds_dify_segment_budget():
    """Chunk vượt max_tokens=500 sẽ bị Dify cắt nhỏ, đoạn sau mất dòng [chunk_id]
    -> parse_chunk_id trả 'unknown' -> mất provenance. Giới hạn ~1200 ký tự tiếng Việt."""
    oversized = [(c.id, len(c.text)) for c in chunk_all_markdown() if len(c.text) > 1200]
    assert not oversized, f"chunk quá lớn, Dify sẽ cắt nhầm: {oversized}"


def test_menu_groups_are_split_into_separate_chunks():
    """Mỗi nhóm món (A. Cà Phê, B. Trà, ...) phải là chunk riêng để truy vấn
    'món nào không có cà phê' không kéo về toàn bộ thực đơn."""
    chunks = chunk_markdown_file(config.KB_DIR / "02_Menu_Va_Phuong_Phap_Pha_Che.md")
    texts = [c.text for c in chunks]
    coffee = next(t for t in texts if "Cà Phê (Coffee)" in t)
    assert "Trà Hoa Nhài" not in coffee, "nhóm Cà Phê không được lẫn nhóm Trà"
    # toppings vẫn phải còn, không được rơi mất khi chẻ nhỏ
    assert any("Extra Shot" in t for t in texts)


def test_subsection_chunks_keep_parent_section_context():
    chunks = chunk_markdown_file(config.KB_DIR / "02_Menu_Va_Phuong_Phap_Pha_Che.md")
    coffee = next(c for c in chunks if "Cà Phê (Coffee)" in c.text)
    assert "VIVA RESERVE" in coffee.text.upper(), "chunk con vẫn phải mang tên quán"
