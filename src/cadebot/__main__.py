"""Điểm vào: `python -m cadebot`. Port lấy từ env để container linh hoạt hơn."""
import os

import uvicorn

from cadebot.api import app


def main() -> None:
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
