"""
从 PDF 字节流提取纯文本（供 Prompt 参考知识使用）。
"""
from io import BytesIO
from typing import Tuple

# 避免单次请求过大，可按需调整
MAX_PDF_TEXT_CHARS = 120000


def extract_text_from_pdf_bytes(data: bytes) -> Tuple[str, str]:
    """
    从 PDF 二进制内容提取文本。

    返回:
        (text, error_message) — 成功时 error_message 为空字符串；失败时 text 为空，error_message 为说明。
    """
    if not data:
        return "", "文件为空"
    try:
        from pypdf import PdfReader
    except ImportError:
        return "", "未安装 pypdf，请执行: pip install pypdf"

    try:
        reader = PdfReader(BytesIO(data))
        parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t.strip())
        text = "\n\n".join(parts)
        text = text.strip()
        if not text:
            return "", "未能从 PDF 中解析出文本（可能是扫描件或加密文件）"
        if len(text) > MAX_PDF_TEXT_CHARS:
            text = text[:MAX_PDF_TEXT_CHARS] + "\n\n[... 文本过长，已截断；可缩小 PDF 或拆分上传 ...]"
        return text, ""
    except Exception as e:
        return "", f"解析 PDF 失败: {e}"
