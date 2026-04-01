from pypdf import PdfReader
import fitz
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from pypdf import PdfReader
def load_pdf_text(pdf_path):
    text = ""

    try:
        pdf_reader = PdfReader(pdf_path)

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    except Exception as e:
        print("PDF extraction failed:", e)

    return text


def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()