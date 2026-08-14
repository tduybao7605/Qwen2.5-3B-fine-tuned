.PHONY: up down restart build logs ps health tunnel-status tunnel-restart tunnel-logs clean

# ── Cadebot API (docker compose) ─────────────────────────────────────────
up: ## Build (nếu cần) và chạy cadebot-api ở nền
	docker compose up -d --build

down: ## Dừng và gỡ container cadebot-api
	docker compose down

restart: down up ## Restart cadebot-api

build: ## Chỉ build lại image, không chạy
	docker compose build

logs: ## Xem log cadebot-api (Ctrl+C để thoát, không dừng container)
	docker compose logs -f cadebot-api

ps: ## Trạng thái container
	docker compose ps

health: ## Gọi thử /health qua cả localhost lẫn domain public
	@echo "── localhost:8000 ──"
	@curl -s http://localhost:8000/health || echo "(không phản hồi)"
	@echo ""
	@echo "── duybao.tdbao-brian.work ──"
	@curl -s https://duybao.tdbao-brian.work/health || echo "(không phản hồi)"
	@echo ""

# ── Cloudflare Tunnel (systemd) ──────────────────────────────────────────
tunnel-status: ## Trạng thái service cloudflared
	systemctl status cloudflared --no-pager

tunnel-restart: ## Restart tunnel (cần sudo)
	sudo systemctl restart cloudflared

tunnel-logs: ## Xem log tunnel (cần sudo)
	sudo journalctl -u cloudflared -f

# ── Dọn dẹp ───────────────────────────────────────────────────────────────
clean: down ## Dừng container + xoá luôn volume hf_cache (model sẽ phải tải lại)
	docker compose down -v
