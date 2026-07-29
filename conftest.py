"""Đảm bảo gốc repo nằm trên sys.path để `import rag` chạy được khi gọi pytest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
