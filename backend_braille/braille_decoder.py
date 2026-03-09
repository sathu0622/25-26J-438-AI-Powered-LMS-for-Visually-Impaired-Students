import cv2
import numpy as np
import fitz 
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Braille mappings
_DOTS = {
    'a':(1,),'b':(1,2),'c':(1,4),'d':(1,4,5),'e':(1,5),
    'f':(1,2,4),'g':(1,2,4,5),'h':(1,2,5),'i':(2,4),'j':(2,4,5),
    'k':(1,3),'l':(1,2,3),'m':(1,3,4),'n':(1,3,4,5),'o':(1,3,5),
    'p':(1,2,3,4),'q':(1,2,3,4,5),'r':(1,2,3,5),'s':(2,3,4),'t':(2,3,4,5),
    'u':(1,3,6),'v':(1,2,3,6),'w':(2,4,5,6),'x':(1,3,4,6),
    'y':(1,3,4,5,6),'z':(1,3,5,6),
    'CAP':(6,),'NUM':(3,4,5,6),
    ',':(2,),';':(2,3),':':(2,5),'.':(2,5,6),
    '?':(2,3,5,6),'!':(2,3,5),'-':(3,6),"'":(4,),' ':(),
}
NUM_MAP = {'a':'1','b':'2','c':'3','d':'4','e':'5',
           'f':'6','g':'7','h':'8','i':'9','j':'0'}


def dots_to_uni(dots):
    c = 0
    for d in dots:
        if 1 <= d <= 6:
            c |= (1 << (d - 1))
    return chr(0x2800 + c)


UNI2ENG = {}
for _e, _d in _DOTS.items():
    UNI2ENG[dots_to_uni(_d)] = _e


def unicode_to_english(text):
    out = []; cap_next = False; cap_word = False
    num = False; prev_was_cap = False
    i = 0; chars = list(text)
    while i < len(chars):
        ch = chars[i]
        if ch in ('\n', '\r'):
            out.append('\n'); num = False; cap_word = False; i += 1; continue
        if ch in (' ', '\u2800'):
            out.append(' '); num = False; cap_word = False; i += 1; continue
        if '\u2801' <= ch <= '\u28FF':
            m = UNI2ENG.get(ch)
            if m is None: i += 1; continue
            if m == 'CAP':
                if prev_was_cap: cap_word = True; cap_next = False; prev_was_cap = False
                else: cap_next = True; prev_was_cap = True
                i += 1; continue
            if m == 'NUM':
                num = True; cap_next = False; cap_word = False; prev_was_cap = False
                i += 1; continue
            prev_was_cap = False
            if num:
                if m in 'abcdefghij': out.append(NUM_MAP[m]); i += 1; continue
                elif not m.isdigit(): num = False
            if (cap_next or cap_word) and m.isalpha():
                m = m.upper(); cap_next = False
            out.append(m)
        else:
            out.append(ch); prev_was_cap = False
        i += 1
    return ''.join(out)


# Image processing helpers
def cluster_1d(vals, gap): #Group numbers into clusters based on proximity; return cluster centers
    if not vals: return []
    s = sorted(vals); g = [[s[0]]]
    for v in s[1:]:
        if v - g[-1][-1] < gap: g[-1].append(v)
        else: g.append([v])
    return [int(np.mean(x)) for x in g]


def any_near(dot_set, cx, cy, r):
    for (dx, dy) in dot_set:
        if abs(dx - cx) <= r and abs(dy - cy) <= r:
            return True
    return False


def pair_columns(col_xs, intra, tol=6):
    xs = sorted(col_xs); cells = []; used = [False] * len(xs); i = 0
    while i < len(xs):
        if used[i]: i += 1; continue
        lx = xs[i]; found = False
        for j in range(i + 1, min(i + 4, len(xs))):
            if not used[j] and abs(xs[j] - lx - intra) <= tol:
                cells.append((float(lx), float(xs[j])))
                used[i] = used[j] = True; found = True; break
        if not found:
            cells.append((float(lx), float(lx + intra))); used[i] = True
        i += 1
    return cells


def fix_first_cell(cells, col_xs, row_dots, row_ys, intra, rad):
    if not cells or not col_xs: return cells
    first_x = col_xs[0]
    has_left  = any(abs(x - (first_x - intra)) <= 6 for x in col_xs)
    has_right = any(abs(x - (first_x + intra)) <= 6 for x in col_xs)
    if has_left or has_right: return cells
    first_col_dots = [(dx, dy) for (dx, dy) in row_dots if abs(dx - first_x) <= rad]
    has_dot6  = any(abs(dy - row_ys[2]) <= rad for (_, dy) in first_col_dots)
    left_dots = any(abs(dy - row_ys[0]) <= rad or abs(dy - row_ys[1]) <= rad
                    for (_, dy) in first_col_dots)
    if has_dot6 and not left_dots:
        cells[0] = (float(first_x - intra), float(first_x))
    return cells


def pil_image_to_english(pil_img):
    # Convert a single PIL image of a Braille page to English text
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY) #Convert to grayscale
    h, w = img.shape
    if w > 2500:
        s = 2500 / w
        img = cv2.resize(img, (int(w * s), int(h * s)))

    blur = cv2.GaussianBlur(img, (5, 5), 0) #Make Braille dots clearer
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) #converts the image into black and white.
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern) #Small noise removed Braille dots remain

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 8]
    if not areas:
        return "[No dots detected on this page]"

    med = np.median(areas)
    dot_set = set()
    for c in cnts:
        a = cv2.contourArea(c)
        if 0.2 * med < a < 4.0 * med: #filter out contours that are too small or too large to be Braille dots
            M = cv2.moments(c)
            if M['m00'] > 0:
                dot_set.add((int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])))

    if not dot_set:
        return "[No valid dots found]"
    dots = list(dot_set) #All Braille dot coordinates

    pts = np.array(dots, dtype=float)
    nn = []
    for i, p in enumerate(pts):
        d = np.sqrt(((pts - p) ** 2).sum(1)); d[i] = np.inf; nn.append(d.min())
    dot_sp = float(np.median(nn))

    pairs_h = [] #distance between left and right columns
    for i, (cx, cy) in enumerate(dots):
        for j, (dx, dy) in enumerate(dots):
            if i != j and abs(dy - cy) < dot_sp * 0.4:
                g = dx - cx
                if dot_sp * 0.5 < g < dot_sp * 1.5:
                    pairs_h.append(g)
    intra = float(np.median(pairs_h)) if pairs_h else dot_sp #column spacing inside a Braille cell

    rad = intra * 0.55
    WORD_GAP = intra * 3.5

    row_centers = cluster_1d([p[1] for p in dots], dot_sp * 0.75) #groups dots into rows
    n_dot_rows = len(row_centers)
    n_txt = n_dot_rows // 3
    if n_txt == 0:
        return "[Could not detect Braille rows]"

    lines = []
    for br in range(n_txt):
        row_ys = [row_centers[br * 3 + di] for di in range(3)]
        y_lo = row_ys[0] - dot_sp; y_hi = row_ys[2] + dot_sp
        row_dots = [(dx, dy) for (dx, dy) in dot_set if y_lo <= dy <= y_hi] #All dots in the current text row
        if not row_dots: lines.append(''); continue

        col_xs = cluster_1d([p[0] for p in row_dots], gap=intra * 0.5)  #Groups dots into columns
        cells  = pair_columns(col_xs, intra, tol=intra * 0.35)
        cells  = fix_first_cell(cells, col_xs, row_dots, row_ys, intra, rad)

        row_chars = []
        for ki, (lx, rx) in enumerate(cells):
            if ki > 0:
                if cells[ki][0] - cells[ki - 1][1] > WORD_GAP:  #Add space between words
                    row_chars.append(dots_to_uni(()))
            active = []
            for di in range(3):
                ry = row_ys[di]
                if any_near(dot_set, lx, ry, rad): active.append(di + 1)
                if any_near(dot_set, rx, ry, rad): active.append(di + 4)
            row_chars.append(dots_to_uni(tuple(sorted(set(active))))) #Convert active dots to Unicode Braille character
        lines.append(''.join(row_chars))

    return unicode_to_english('\n'.join(lines))


def decode_pdf(pdf_bytes: bytes) -> list[str]:
    # Convert PDF bytes → list of decoded text strings, one per page.
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = []
    for page_num, page in enumerate(doc, start=1):
        try:
            pix = page.get_pixmap(dpi=200) #converts the PDF page into an image
            pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples) #converts the pixmap image into a PIL image.
            text = pil_image_to_english(pil_img)
        except Exception as e:
            text = f"[Error on page {page_num}: {str(e)}]"
        all_text.append(text)
    return all_text