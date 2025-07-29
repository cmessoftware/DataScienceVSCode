import argparse
import sys
import nbformat
from nbconvert import PDFExporter
import subprocess

def convert_to_pdf(ipynb_path, output_path=None):
    # Load the notebook
    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Convert to PDF
    pdf_exporter = PDFExporter()
    pdf_data, _ = pdf_exporter.from_notebook_node(notebook)

    # Define output path
    if not output_path:
        output_path = ipynb_path.replace(".ipynb", ".pdf")

    # Save PDF
    with open(output_path, "wb") as f:
        f.write(pdf_data)

    print(f"✅ PDF saved to: {output_path}")
    
def convert_to_docx(ipynb_path, output_path=None):
    # Step 1: Convert .ipynb to .md using nbconvert
    subprocess.run(["jupyter", "nbconvert", "--to", "markdown", ipynb_path], check=True)

    # Step 2: Find generated .md
    md_path = ipynb_path.replace(".ipynb", ".md")
    if not output_path:
        output_path = ipynb_path.replace(".ipynb", ".docx")

    # Step 3: Convert .md to .docx using pandoc
    subprocess.run(["pandoc", md_path, "-o", output_path], check=True)

    print(f"✅ .docx saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .ipynb to .pdf / .docx")
    parser.add_argument("notebook", help="Path to the .ipynb file")
    parser.add_argument("-d", "--docx", action="store_true", help="Convert to .docx instead of .pdf")
    parser.add_argument("-n", "--notebook", help="Path to the .ipynb file")
    parser.add_argument("-o", "--output", help="Optional output path for the PDF")
    args = parser.parse_args()
    
    if args.docx:
        convert_to_docx(args.notebook, args.output)
    else: 
        convert_to_pdf(args.notebook, args.output)
