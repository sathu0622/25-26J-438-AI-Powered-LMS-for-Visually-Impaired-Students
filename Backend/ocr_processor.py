# ocr_processor.py
import os
import tempfile
import cv2
import pytesseract
import pdf2image
from pdf2image import convert_from_path
from PIL import Image
import numpy as np

def ocr_image_simple(image_path: str) -> str:
    """
    Simple OCR function using basic Tesseract.
    """
    try:
        print(f"Processing image: {image_path}")
        
        # Load image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"  Warning: Could not read image with OpenCV, trying PIL fallback...")
            # Fallback to PIL
            pil_img = Image.open(image_path)
            # Convert PIL image to numpy array
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        height, width = img.shape[:2]
        print(f"  Image size: {width}x{height}")
        
        # Convert to RGB (Tesseract needs RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run OCR with standard configuration
        text = pytesseract.image_to_string(img_rgb, lang="eng")
        
        print(f"  Extracted {len(text)} characters")
        return text.strip()
        
    except Exception as e:
        print(f"OCR Error: {e}")
        # Try alternative method
        try:
            # Direct PIL approach
            pil_img = Image.open(image_path)
            text = pytesseract.image_to_string(pil_img, lang="eng")
            return text.strip()
        except Exception as e2:
            print(f"Fallback OCR also failed: {e2}")
            return ""

def ocr_pdf_simple(pdf_path: str) -> str:
    """Extract text from PDF using simple OCR on all pages."""
    try:
        print(f"Processing PDF: {pdf_path}")
        
        # Convert PDF to images with standard DPI
        print("  Converting PDF to images...")
        pages = convert_from_path(pdf_path, dpi=200)
        
        print(f"  Converted {len(pages)} pages")
        
        all_pages_text = []
        
        for page_num, page in enumerate(pages):
            print(f"  Processing page {page_num + 1}/{len(pages)}...")
            
            # Save page as temporary image
            temp_dir = tempfile.gettempdir()
            temp_img = os.path.join(temp_dir, f"temp_page_{page_num}_{os.urandom(4).hex()}.png")
            
            page.save(temp_img, "PNG", dpi=(200, 200))
            
            # Process with simple OCR
            page_text = ocr_image_simple(temp_img)
            
            if page_text.strip():
                all_pages_text.append(page_text)
            
            # Clean up
            try:
                os.remove(temp_img)
            except:
                pass
        
        # Combine all pages with page break markers
        combined = "\n\n[PAGE BREAK]\n\n".join(all_pages_text)
        print(f"  Total extracted characters: {len(combined)}")
        
        return combined
        
    except Exception as e:
        print(f"PDF OCR Error: {e}")
        # Try to get at least some text
        try:
            # Try with first page only
            pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200)
            if pages:
                temp_dir = tempfile.gettempdir()
                temp_img = os.path.join(temp_dir, f"temp_fallback_{os.urandom(4).hex()}.png")
                pages[0].save(temp_img, "PNG", dpi=(200, 200))
                text = ocr_image_simple(temp_img)
                try:
                    os.remove(temp_img)
                except:
                    pass
                return text
        except:
            pass
        return ""

def extract_text_book(input_path: str) -> str:
    """Extract text for books using simple OCR."""
    if input_path.lower().endswith(".pdf"):
        return ocr_pdf_simple(input_path)
    else:
        return ocr_image_simple(input_path)