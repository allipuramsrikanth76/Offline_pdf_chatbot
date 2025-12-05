from pypdf import PdfReader

def load_pdf_text(path: str):
    """Extract all text from a PDF file and return as one string."""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)

if __name__=="__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <file.pdf>")
        sys.exit(1)
    text = load_pdf_text(sys.argv[1])
    print(text[:1000])
