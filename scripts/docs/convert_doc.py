#!/usr/bin/env python3
"""
Skrypt do konwersji pliku .doc do .docx przy użyciu COM automation.
"""

import sys
from pathlib import Path

def convert_doc_to_docx(doc_path: Path, docx_path: Path):
    """Konwertuje plik .doc do .docx."""
    import win32com.client as win32

    word = win32.Dispatch("Word.Application")
    word.Visible = False

    try:
        doc = word.Documents.Open(str(doc_path.absolute()))
        doc.SaveAs2(str(docx_path.absolute()), FileFormat=16)  # 16 = docx format
        doc.Close()
        print(f"Skonwertowano: {docx_path}")
    finally:
        word.Quit()


def main():
    project_root = Path(__file__).parent.parent.parent

    # Konwertuj szablon DTP
    doc_path = project_root / "template" / "PG_WETI_DTP_wer. 1.00.doc"
    docx_path = project_root / "template" / "PG_WETI_DTP_wer. 1.00.docx"

    if not doc_path.exists():
        print(f"Błąd: Nie znaleziono pliku {doc_path}")
        return

    convert_doc_to_docx(doc_path, docx_path)


if __name__ == "__main__":
    main()
