from __future__ import annotations

from pdf2image import convert_from_bytes

f_data = None
with open("zayavlenie1.pdf", "rb") as f:
    f_data = f.read()
images = convert_from_bytes(f_data)
for img in images:
    img.save(r"new_folder\output.jpg", "JPEG")
