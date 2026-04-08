# extract_text
from pypdf import PdfReader
import os

def extract_pdf_text(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_txt_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def load_all_documents(data_dir="data"):
    all_text = ""
    
    # Load resume PDF
    pdf_path = os.path.join(data_dir, "resume.pdf")
    if os.path.exists(pdf_path):
        print("Extracting resume PDF...")
        all_text += "=== RESUME ===\n"
        all_text += extract_pdf_text(pdf_path)
        all_text += "\n\n"
    
    # Load GitHub projects text
    txt_path = os.path.join(data_dir, "github_projects.txt")
    if os.path.exists(txt_path):
        print("Loading GitHub projects...")
        all_text += "=== GITHUB PROJECTS ===\n"
        all_text += extract_txt_file(txt_path)
    
    return all_text

if __name__ == "__main__":
    text = load_all_documents()
    print(f"\nTotal characters extracted: {len(text)}")
    print("\nFirst 5000 characters preview:")
    print(text[:500])
    
    # Save combined text for inspection
    with open("data/combined_text.txt", "w") as f:
        f.write(text)
    print("\nSaved to data/combined_text.txt")