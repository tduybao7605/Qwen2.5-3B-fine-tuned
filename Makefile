.PHONY: help up down restart build logs ps health tunnel-status tunnel-restart tunnel-logs clean test

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Cadebot API (docker compose) ─────────────────────────────────────────
up: ## Build if needed and run cadebot-api in the background
	docker compose up -d --build

down: ## Stop and remove the cadebot-api container
	docker compose down

restart: down up ## Restart cadebot-api

build: ## Rebuild the image only, without starting it
	docker compose build

logs: ## Follow cadebot-api logs (Ctrl+C detaches, does not stop the container)
	docker compose logs -f cadebot-api

ps: ## Show container status
	docker compose ps

health: ## Probe /health locally, and at PUBLIC_URL if it is set
	@echo "── localhost:8000 ──"
	@curl -s http://localhost:8000/health || echo "(no response)"
	@echo ""
	@if [ -n "$$PUBLIC_URL" ]; then \
		echo "── $$PUBLIC_URL ──"; \
		curl -s "$$PUBLIC_URL/health" || echo "(no response)"; \
		echo ""; \
	else \
		echo "(set PUBLIC_URL=https://your-tunnel.example.com to also probe the public endpoint)"; \
	fi

# ── Cloudflare Tunnel (systemd) ──────────────────────────────────────────
tunnel-status: ## Show the cloudflared service status
	systemctl status cloudflared --no-pager

tunnel-restart: ## Restart the tunnel (needs sudo)
	sudo systemctl restart cloudflared

tunnel-logs: ## Follow tunnel logs (needs sudo)
	sudo journalctl -u cloudflared -f

# ── Development ──────────────────────────────────────────────────────────
test: ## Run the test suite
	python3 -m pytest tests/ -q

clean: down ## Stop the container and delete the hf_cache volume (forces a full model re-download)
	docker compose down -v
