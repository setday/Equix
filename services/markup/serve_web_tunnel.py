"""Markup service using improved architecture."""

from __future__ import annotations

from src.services.layout_service import create_layout_service

# Create service instance
service = create_layout_service()

if __name__ == "__main__":
    service.run(host="0.0.0.0", port=5123)
