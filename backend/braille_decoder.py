# braille_decoder.py

import cv2
import numpy as np

# -----------------------
# Braille Dictionary
# -----------------------
UNICODE_BRAILLE_TO_TEXT = {
    '⠁': 'a', '⠃': 'b', '⠉': 'c', '⠙': 'd', '⠑': 'e',
    '⠋': 'f', '⠛': 'g', '⠓': 'h', '⠊': 'i', '⠚': 'j',
    '⠅': 'k', '⠇': 'l', '⠍': 'm', '⠝': 'n', '⠕': 'o',
    '⠏': 'p', '⠟': 'q', '⠗': 'r', '⠎': 's', '⠞': 't',
    '⠥': 'u', '⠧': 'v', '⠺': 'w', '⠭': 'x', '⠽': 'y',
    '⠵': 'z', ' ': ' '
}

# -----------------------
# Dot Detection
# -----------------------
def detect_dots(img_path, debug=False):
    """Detect all Braille dots in the image"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Detect contours (dots)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 3 < w < 30 and 3 < h < 30:  # adjust for your dot size
            cx, cy = x + w // 2, y + h // 2
            dots.append((cx, cy))
    
    return sorted(dots, key=lambda p: (p[1], p[0]))

# -----------------------
# Group into Lines
# -----------------------
def group_into_lines(dots):
    """Group dots into horizontal lines based on Y-coordinate"""
    if not dots:
        return []
    
    dots_sorted = sorted(dots, key=lambda p: p[1])
    lines = []
    current_line = [dots_sorted[0]]

    for i in range(1, len(dots_sorted)):
        if dots_sorted[i][1] - dots_sorted[i-1][1] > 20:  # line threshold
            lines.append(current_line)
            current_line = []
        current_line.append(dots_sorted[i])
    
    if current_line:
        lines.append(current_line)
    
    return lines

# -----------------------
# Group Line into Cells
# -----------------------
def group_into_cells(line):
    """Group line dots into Braille cells using X-coordinate"""
    if not line:
        return []
    
    line = sorted(line, key=lambda p: p[0])
    cells = []
    current_cell = []
    last_x = line[0][0]

    for dot in line:
        if dot[0] - last_x > 18:  # horizontal gap threshold
            cells.append(current_cell)
            current_cell = []
        current_cell.append(dot)
        last_x = dot[0]
    
    if current_cell:
        cells.append(current_cell)
    
    return cells

# -----------------------
# Cell → 6-dot Pattern
# -----------------------
def cell_to_pattern(cell):
    """
    Convert a Braille cell to its 6-dot pattern.
    Dot positions:
    1 4
    2 5
    3 6
    """
    if not cell:
        return "000000"
    
    xs = [p[0] for p in cell]
    ys = [p[1] for p in cell]
    x_thresh = np.median(xs)

    left = sorted([p for p in cell if p[0] <= x_thresh], key=lambda p: p[1])
    right = sorted([p for p in cell if p[0] > x_thresh], key=lambda p: p[1])

    pattern = ["0"] * 6
    for i in range(min(3, len(left))):
        pattern[i] = "1"
    for i in range(min(3, len(right))):
        pattern[i + 3] = "1"
    
    return "".join(pattern)

# -----------------------
# Pattern → Unicode
# -----------------------
def pattern_to_unicode(pattern):
    bits = [int(b) for b in pattern]
    mapping = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20]
    code = sum(b * m for b, m in zip(bits, mapping))
    return chr(0x2800 + code)

# -----------------------
# Unicode → English
# -----------------------
def unicode_to_text(unicode_str):
    return ''.join(UNICODE_BRAILLE_TO_TEXT.get(c, '?') for c in unicode_str)

# -----------------------
# Main Function
# -----------------------
def braille_image_to_text(img_path, debug=False):
    """Convert Braille image → Unicode → English text"""
    dots = detect_dots(img_path, debug=debug)
    if not dots:
        return ""
    
    lines = group_into_lines(dots)
    result = []

    for line in lines:
        cells = group_into_cells(line)
        unicode_line = ""
        for cell in cells:
            pattern = cell_to_pattern(cell)
            unicode_char = pattern_to_unicode(pattern)
            unicode_line += unicode_char
        
        english_line = unicode_to_text(unicode_line)
        result.append(english_line)
    
    return "\n".join(result)

# -----------------------
# Test
# -----------------------
if __name__ == "__main__":
    img_path = "braille_sample.png"
    text = braille_image_to_text(img_path, debug=True)
    print("Detected English text:\n", text)
