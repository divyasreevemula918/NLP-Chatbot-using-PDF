from pypdf import PdfReader
import fitz
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def load_pdf_text(pdf_path):
    text = ""

    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        print("Normal extraction result length:", len(text))
    except Exception as e:
        print("Normal PDF extraction failed:", e)

    if text.strip():
        print("Using normal PDF text extraction")
        return text

    print("No selectable text found. Trying OCR...")

    ocr_text = ""

    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            page_text = pytesseract.image_to_string(img)
            print(f"OCR page {page_num + 1} length:", len(page_text))

            if page_text:
                ocr_text += page_text + "\n"

        doc.close()
    except Exception as e:
        print("OCR extraction failed:", e)

    print("Final OCR result length:", len(ocr_text))
    return ocr_text


def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()