-- CẤU TRÚC VÀ DỮ LIỆU POSTGRESQL CHUẨN ĐƯỢC NẠP TỪ CADEBOT UI (app-debug.apk)

-- 1. Bảng thực đơn đồ uống & món ăn
CREATE TABLE IF NOT EXISTS menu_items (
    id SERIAL PRIMARY KEY,
    item_code VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL,
    price INT NOT NULL,
    description TEXT,
    brewing_method VARCHAR(50),
    is_available BOOLEAN DEFAULT TRUE,
    tags TEXT,
    attributes JSONB,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Bảng chương trình khuyến mãi & Combo
CREATE TABLE IF NOT EXISTS promotions (
    id SERIAL PRIMARY KEY,
    promo_code VARCHAR(50) UNIQUE NOT NULL,
    title VARCHAR(150) NOT NULL,
    discount_detail TEXT NOT NULL,
    start_date DATE,
    end_date DATE,
    conditions TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Bảng bộ câu hỏi thường gặp (FAQ)
CREATE TABLE IF NOT EXISTS faqs (
    id SERIAL PRIMARY KEY,
    faq_id VARCHAR(50) UNIQUE NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    related_items TEXT,
    tags TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NẠP DỮ LIỆU THỰC ĐƠN TỪ APK UI CADEBOT
INSERT INTO menu_items (item_code, name, category, price, description, brewing_method, is_available, tags) VALUES
('VR_LATTE_M', 'Viva Latte', 'Cà Phê', 55000, 'Latte signature của Viva, vị sữa béo nhẹ và espresso cân bằng.', 'Pha Máy', TRUE, 'best_seller, milk, coffee'),
('VR_CAPPUCCINO', 'Cappuccino', 'Cà Phê', 50000, 'Cappuccino cổ điển với lớp bọt sữa dày mịn, hương espresso đậm đà.', 'Pha Máy', TRUE, 'classic, milk, coffee'),
('VR_AMERICANO', 'Americano', 'Cà Phê', 45000, 'Espresso pha loãng với nước nóng, thanh thoát và nhẹ nhàng.', 'Pha Máy', TRUE, 'classic, black, coffee'),
('VR_MATCHA_LATTE', 'Matcha Latte', 'Trà', 60000, 'Matcha Nhật Bản cao cấp hòa quyện cùng sữa tươi thơm béo.', 'Pha Trực Tiếp', TRUE, 'tea, milk, matcha'),
('VR_JASMINE_TEA', 'Trà Hoa Nhài', 'Trà', 40000, 'Trà hoa nhài thanh mát, hương thơm tự nhiên dịu nhẹ.', 'Pha Trực Tiếp', TRUE, 'tea, refreshing, no_coffee'),
('VR_PEACH_TEA', 'Trà Đào Cam Sả', 'Trà', 45000, 'Kết hợp đào, cam và sả tươi, vừa thơm vừa mát lạnh.', 'Pha Trực Tiếp', TRUE, 'tea, fruity, refreshing'),
('VR_STRAWBERRY_BLEND', 'Dâu Đá Xay', 'Đá Xay', 65000, 'Đá xay dâu tây tươi, mịn như kem, ngọt ngào và bắt mắt.', 'Đá Xay', TRUE, 'ice_blended, fruity, sweet'),
('VR_MOCHA_BLEND', 'Mocha Đá Xay', 'Đá Xay', 65000, 'Espresso và sô cô la xay cùng đá, phủ kem tươi béo ngậy.', 'Đá Xay', TRUE, 'ice_blended, coffee, chocolate'),
('VR_CROISSANT', 'Bánh Croissant Bơ', 'Bánh Ngọt', 35000, 'Croissant bơ kiểu Pháp, vỏ giòn xốp, ruột mềm thơm lừng.', 'Pha Trực Tiếp', TRUE, 'pastry, french, butter'),
('VR_TIRAMISU', 'Bánh Tiramisu', 'Bánh Ngọt', 55000, 'Tiramisu Ý thơm nức với mascarpone, cà phê và cacao nguyên chất.', 'Pha Trực Tiếp', TRUE, 'pastry, italian, dessert'),
('VR_COMBO_A', 'Combo Sáng Sớm', 'Combo', 85000, '1 Americano + 1 Croissant Bơ. Tiết kiệm 5.000đ so với gọi riêng.', 'Đá Xay', TRUE, 'combo, morning, deal'),
('VR_COMBO_B', 'Combo Chiều Thư Giãn', 'Combo', 110000, '1 Matcha Latte + 1 Dâu Đá Xay + 1 Bánh Tiramisu. Ưu đãi đặc biệt.', 'Đá Xay', TRUE, 'combo, afternoon, deal')
ON CONFLICT (item_code) DO NOTHING;

-- NẠP DỮ LIỆU KHUYẾN MÃI TỪ APK UI CADEBOT
INSERT INTO promotions (promo_code, title, discount_detail, start_date, end_date, conditions) VALUES
('camp_001', 'Combo Sáng Sớm', 'Americano + Croissant chỉ 85.000đ - Khởi đầu ngày mới năng lượng với combo tiết kiệm 5.000đ.', '2026-06-01', '2026-12-31', 'Áp dụng cho khách hàng tại Viva Reserve'),
('camp_002', 'Chiều Thư Giãn', 'Matcha + Dâu Đá Xay + Tiramisu 110.000đ - Bộ 3 hoàn hảo cho chiều tà, tiết kiệm hơn 10.000đ.', '2026-06-01', '2026-12-31', 'Áp dụng cho khách hàng tại Viva Reserve'),
('camp_003', 'Viva Reserve Experience', 'Khám phá không gian cà phê độc đáo - Cadebot L100 phục vụ tận tay — trải nghiệm cà phê tương lai tại Viva Reserve.', '2026-06-01', '2026-12-31', 'Áp dụng cho khách hàng tại Viva Reserve')
ON CONFLICT (promo_code) DO NOTHING;

-- NẠP DỮ LIỆU BỘ CÂU HỎI FAQ TỪ APK UI CADEBOT
INSERT INTO faqs (faq_id, question, answer, related_items, tags) VALUES
('faq_001', 'Viva Latte có vị như thế nào?', 'Viva Latte có vị sữa béo nhẹ kết hợp espresso cân bằng, không quá đắng cũng không quá ngọt. Đây là món best seller của Viva Reserve.', 'VR_LATTE_M', 'menu_qa, coffee'),
('faq_002', 'Latte có caffeine không?', 'Có, Viva Latte chứa espresso nên có caffeine. Nếu bạn muốn tránh caffeine, mình gợi ý Matcha Latte hoặc Trà Đào Cam Sả — đều không có cà phê.', 'VR_LATTE_M, VR_MATCHA_LATTE, VR_PEACH_TEA', 'product_qa, caffeine'),
('faq_003', 'Món nào không có cà phê?', 'Các món không chứa cà phê tại Viva: Matcha Latte, Trà Hoa Nhài, Trà Đào Cam Sả, Dâu Đá Xay, Croissant Bơ. Bạn muốn thử món nào?', 'VR_MATCHA_LATTE, VR_JASMINE_TEA, VR_PEACH_TEA, VR_STRAWBERRY_BLEND', 'recommendation, no_coffee'),
('faq_004', 'Có món nào ít ngọt không?', 'Bạn có thể chọn độ ngọt 0% hoặc 30% cho hầu hết các món. Americano và Trà Hoa Nhài rất phù hợp nếu bạn thích ít ngọt tự nhiên.', 'VR_AMERICANO, VR_JASMINE_TEA', 'recommendation, less_sweet'),
('faq_005', 'Gọi món như thế nào?', 'Bạn có thể chọn món trực tiếp từ menu trên màn hình, hoặc nói với mình để mình gợi ý và thêm vào giỏ hàng. Sau khi xác nhận giỏ hàng, bạn quét mã QR để thanh toán.', '', 'how_to_order'),
('faq_006', 'Thanh toán bằng gì?', 'Tại Viva Reserve, bạn thanh toán bằng mã QR (MoMo, VNPay, VietQR) hoặc tiền mặt qua nhân viên. Robot sẽ hiển thị mã QR sau khi bạn xác nhận giỏ hàng.', '', 'payment, how_to'),
('faq_007', 'Robot có giao món đến bàn không?', 'Có! Sau khi thanh toán, nhân viên sẽ chuẩn bị đồ uống và robot sẽ giao đến tận bàn của bạn. Màn hình sẽ thông báo khi robot đang đến.', '', 'delivery, robot'),
('faq_008', 'Matcha Latte có sữa không?', 'Có, Matcha Latte dùng sữa tươi thơm béo. Nếu bạn không dùng được sữa bò, có thể chọn topping Oat Milk để thay thế.', 'VR_MATCHA_LATTE', 'product_qa, milk, allergen'),
('faq_009', 'Combo nào tiết kiệm nhất?', 'Combo Sáng Sớm (Americano + Croissant) giá 85.000đ, tiết kiệm 5.000đ. Combo Chiều Thư Giãn (Matcha Latte + Dâu Đá Xay + Tiramisu) giá 110.000đ, tiết kiệm hơn 10.000đ so với gọi riêng.', 'VR_COMBO_A, VR_COMBO_B', 'promotion, combo, value'),
('faq_010', 'Có ưu đãi gì không?', 'Hiện tại Viva có Combo Sáng Sớm và Combo Chiều Thư Giãn với giá ưu đãi. Bạn có thể hỏi nhân viên về các chương trình khuyến mãi mới nhất.', 'VR_COMBO_A, VR_COMBO_B', 'promotion'),
('faq_011', 'Croissant có vị gì?', 'Croissant Bơ của Viva kiểu Pháp, vỏ ngoài giòn xốp nhiều lớp, bên trong mềm thơm mùi bơ. Phù hợp ăn kèm cà phê buổi sáng.', 'VR_CROISSANT', 'product_qa, pastry'),
('faq_012', 'Dâu Đá Xay có ngọt không?', 'Dâu Đá Xay mặc định 70% đường, khá ngọt và thơm mùi dâu tươi. Bạn có thể chọn 50% đường nếu muốn vừa ngọt hơn.', 'VR_STRAWBERRY_BLEND', 'product_qa, sweetness'),
('faq_013', 'Gọi thêm món được không?', 'Được! Bạn nhấn nút Bắt đầu đặt món hoặc nói với mình để thêm món. Nhân viên cũng hỗ trợ nếu bạn cần.', '', 'how_to_order'),
('faq_014', 'Gọi thêm món kèm thì có những gì?', 'Quán có các loại topping gọi kèm đồ uống như Extra Shot espresso, Sữa yến mạch (Oat Milk), Trân châu, Thạch, Kem tươi, Sốt sô cô la, Sốt dâu. Ngoài ra bạn có thể gọi thêm bánh ngọt ăn kèm như Bánh Croissant Bơ (35.000đ) hoặc Bánh Tiramisu Ý (55.000đ).', '', 'topping, pastry, side_item'),
('faq_015', 'Trà Đào Cam Sả có vị chua không?', 'Trà Đào Cam Sả có vị chua nhẹ tự nhiên từ cam, kết hợp ngọt thơm của đào và mát của sả. Rất sảng khoái, phù hợp buổi chiều.', 'VR_PEACH_TEA', 'product_qa, tea'),
('faq_016', 'Món nào phù hợp uống nóng?', 'Cappuccino, Americano và Viva Latte đều có option nóng rất ngon. Trà Hoa Nhài nóng cũng rất thơm và dễ uống.', 'VR_CAPPUCCINO, VR_AMERICANO, VR_LATTE_M, VR_JASMINE_TEA', 'recommendation, hot'),
('faq_017', 'Có topping nào không?', 'Viva có các topping: Extra Shot (espresso thêm), Oat Milk (sữa yến mạch), Trân Châu, Thạch, Kem Tươi, Sốt Sô Cô La, Sốt Dâu. Tuỳ từng món sẽ có topping khác nhau.', '', 'product_qa, topping'),
('faq_018', 'Tôi muốn gọi nhân viên', 'Bạn nhấn nút Gọi nhân viên trên màn hình, nhân viên Viva sẽ đến hỗ trợ bạn ngay.', '', 'call_staff'),
('faq_019', 'Size nào phổ biến nhất?', 'Size M (medium) là size phổ biến nhất tại Viva. Nếu bạn muốn nhiều hơn hoặc chia sẻ, size L là lựa chọn tốt.', '', 'product_qa, size'),
('faq_020', 'Tiramisu có cà phê không?', 'Có, Tiramisu Ý chứa espresso và cacao, nên có caffeine. Nếu bạn tránh caffeine, Croissant Bơ và Dâu Đá Xay là lựa chọn không có cà phê.', 'VR_TIRAMISU, VR_CROISSANT, VR_STRAWBERRY_BLEND', 'product_qa, caffeine, pastry'),
('faq_021', 'Bao lâu thì nhận được món?', 'Sau khi thanh toán, nhân viên sẽ pha chế trong khoảng 3-5 phút và robot sẽ giao đến bàn. Màn hình sẽ hiển thị trạng thái đơn hàng của bạn.', '', 'how_to, delivery, time')
ON CONFLICT (faq_id) DO NOTHING;
