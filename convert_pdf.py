import fitz  # PyMuPDF
import os

pdf_path = r"/home/kali/Desktop/307-Mountain Vista at Sunset Flats-Pulte Homes.pdf"
output_dir = r"/home/kali/Desktop/307-Mountain Vista at Sunset Flats-Pulte Homes"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

try:
    doc = fitz.open(pdf_path)
    print(f"Opened PDF: {pdf_path}")
    print(f"Total pages: {len(doc)}")

    for i, page in enumerate(doc):
        # Determine batch folder
        batch_num = (i // 10) + 1
        batch_dir = os.path.join(output_dir, f"Batch_{batch_num}")
        
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
            
        # Render page to an image
        pix = page.get_pixmap()
        output_file = os.path.join(batch_dir, f"page_{i + 1}.jpg")
        pix.save(output_file)
        print(f"Saved: {output_file}")
    
    print("Conversion and grouping complete!")
except Exception as e:
    print(f"Error: {e}")
