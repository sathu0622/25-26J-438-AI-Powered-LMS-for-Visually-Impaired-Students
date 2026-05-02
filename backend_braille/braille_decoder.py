import cv2
import numpy as np
import fitz
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

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
    """Convert a tuple of active dot numbers (1–6) to a Unicode Braille character."""
    c = 0
    for d in dots:
        if 1 <= d <= 6:
            c |= (1 << (d - 1))
    return chr(0x2800 + c)


# Build reverse map: Unicode Braille char → English token
UNI2ENG = {}
for _e, _d in _DOTS.items():
    UNI2ENG[dots_to_uni(_d)] = _e


def unicode_to_english(text):
    """Translate a string of Unicode Braille characters to English."""
    out = []
    cap_next = False
    cap_word = False
    num = False
    prev_was_cap = False
    i = 0
    chars = list(text)

    while i < len(chars):
        ch = chars[i]

        if ch in ('\n', '\r'):
            out.append('\n')
            num = False
            cap_word = False
            i += 1
            continue

        if ch in (' ', '\u2800'):
            out.append(' ')
            num = False
            cap_word = False
            i += 1
            continue

        if '\u2801' <= ch <= '\u28FF':
            m = UNI2ENG.get(ch)
            if m is None:
                i += 1
                continue

            if m == 'CAP':
                if prev_was_cap:
                    cap_word = True
                    cap_next = False
                    prev_was_cap = False
                else:
                    cap_next = True
                    prev_was_cap = True
                i += 1
                continue

            if m == 'NUM':
                num = True
                cap_next = False
                cap_word = False
                prev_was_cap = False
                i += 1
                continue

            prev_was_cap = False

            if num:
                if m in 'abcdefghij':
                    out.append(NUM_MAP[m])
                    i += 1
                    continue
                elif not m.isdigit():
                    num = False

            if (cap_next or cap_word) and m.isalpha():
                m = m.upper()
                cap_next = False

            out.append(m)
        else:
            out.append(ch)
            prev_was_cap = False

        i += 1

    return ''.join(out)



def cluster_1d(vals, gap):
    """Group numbers into clusters based on proximity; return cluster centres."""
    if not vals:
        return []
    s = sorted(vals)
    g = [[s[0]]]
    for v in s[1:]:
        if v - g[-1][-1] < gap:
            g[-1].append(v)
        else:
            g.append([v])
    return [int(np.mean(x)) for x in g]


def cluster_1d_members(vals, gap):
    """Group numbers into clusters; return list of (center, members) tuples."""
    if not vals:
        return []
    s = sorted(vals)
    groups = [[s[0]]]
    for v in s[1:]:
        if v - groups[-1][-1] < gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [(int(np.mean(g)), g) for g in groups]


def any_near(dot_set, cx, cy, r):
    """Return True if any dot in dot_set is within radius r of (cx, cy)."""
    for (dx, dy) in dot_set:
        if abs(dx - cx) <= r and abs(dy - cy) <= r:
            return True
    return False


def pair_columns(col_xs, intra, tol=6):
    """Group detected dot columns into left/right pairs to form Braille cells."""
    xs = sorted(col_xs)
    cells = []
    used = [False] * len(xs)
    i = 0
    while i < len(xs):
        if used[i]:
            i += 1
            continue
        lx = xs[i]
        found = False
        for j in range(i + 1, min(i + 4, len(xs))):
            if not used[j] and abs(xs[j] - lx - intra) <= tol:
                cells.append((float(lx), float(xs[j])))
                used[i] = used[j] = True
                found = True
                break
        if not found:
            cells.append((float(lx), float(lx + intra)))
            used[i] = True
        i += 1
    return cells


def find_3_subrows(subrow_centers, dot_set, v_sp):
    if len(subrow_centers) >= 3:
        return [float(subrow_centers[0]),
                float(subrow_centers[1]),
                float(subrow_centers[2])]

    if len(subrow_centers) == 2:
        c1, c2 = float(subrow_centers[0]), float(subrow_centers[1])
        d = c2 - c1
        if d < v_sp * 1.3:
            # Rows 1 & 2 detected — infer row 3
            return [c1, c2, c2 + v_sp]
        else:
            # Rows 1 & 3 detected — infer row 2 (middle)
            return [c1, (c1 + c2) / 2, c2]

    # Only 1 center detected — recover 3 sub-rows from actual dot spread
    center = float(subrow_centers[0])
    band_ys = sorted([dy for (dx, dy) in dot_set if abs(dy - center) < v_sp * 1.8])
    if not band_ys:
        return [center - v_sp, center, center + v_sp]

    y_min, y_max = band_ys[0], band_ys[-1]
    span = y_max - y_min

    if span < v_sp * 0.8:
        # Dots all in one tight band — use v_sp offsets
        return [center - v_sp, center, center + v_sp]

    # Divide span into 3 equal thirds and take mean of each third
    third = span / 3.0
    r1 = [y for y in band_ys if y <= y_min + third]
    r2 = [y for y in band_ys if y_min + third < y <= y_min + 2 * third]
    r3 = [y for y in band_ys if y > y_min + 2 * third]

    c1 = float(np.mean(r1)) if r1 else y_min
    c2 = float(np.mean(r2)) if r2 else y_min + v_sp
    c3 = float(np.mean(r3)) if r3 else y_min + 2 * v_sp
    return [c1, c2, c3]


# ---------------------------------------------------------------------------
# Core page decoder
# ---------------------------------------------------------------------------

def pil_image_to_english(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    h, w = img.shape

    # Allow larger images so dots have more pixels
    if w > 3000:
        s = 3000 / w
        img = cv2.resize(img, (int(w * s), int(h * s)))

    # FIX 1: Improved preprocessing — larger blur + blockSize + C + kernel
    blur = cv2.GaussianBlur(img, (9, 9), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,
        C=10
    )
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern)

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    MIN_DOT_AREA = 30
    areas = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > MIN_DOT_AREA]
    if not areas:
        return "[No dots detected on this page]"

    med = np.median(areas)
    dot_list = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a > MIN_DOT_AREA and 0.4 * med < a < 2.5 * med:
            M = cv2.moments(c)
            if M['m00'] > 0:
                dot_list.append((int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])))

    if not dot_list:
        return "[No valid dots found]"

    # Remove spatially isolated dots (scanner artefacts with no nearby neighbours)
    pts = np.array(dot_list, dtype=float)
    nn = []
    for i, p in enumerate(pts):
        d = np.sqrt(((pts - p) ** 2).sum(1))
        d[i] = np.inf
        nn.append(d.min())
    dot_sp = float(np.median(nn))

    keep = [dot_list[i] for i, n in enumerate(nn) if n < dot_sp * 4.0]
    dot_set = set(keep)
    dots = list(dot_set)

    if not dots:
        return "[No valid dots after noise filter]"

    # Re-estimate spacing on clean dot set
    pts = np.array(dots, dtype=float)
    nn2 = []
    for i, p in enumerate(pts):
        d = np.sqrt(((pts - p) ** 2).sum(1))
        d[i] = np.inf
        nn2.append(d.min())
    dot_sp = float(np.median(nn2))

    # Intra-cell horizontal column spacing
    pairs_h = []
    for cx, cy in dots:
        for dx, dy in dots:
            if abs(dy - cy) < dot_sp * 0.4:
                g = dx - cx
                if dot_sp * 0.5 < g < dot_sp * 1.5:
                    pairs_h.append(g)
    intra = float(np.median(pairs_h)) if pairs_h else dot_sp

    # Within-cell vertical dot spacing
    pairs_v = []
    for cx, cy in dots:
        for dx, dy in dots:
            if abs(dx - cx) < intra * 0.5:
                g = dy - cy
                if dot_sp * 0.3 < g < dot_sp * 2.0:
                    pairs_v.append(g)
    v_sp = float(np.median(pairs_v)) if pairs_v else dot_sp

    rad = intra * 0.55
    WORD_GAP = intra * 3.5

    # Tight clustering to find individual dot sub-rows (~v_sp apart)
    subrow_clusters = cluster_1d_members([p[1] for p in dots], gap=v_sp * 0.4)

    # Discard sub-row clusters with too few dots — they are scanner noise
    MIN_SUBROW_DOTS = 3
    valid_subrow_ys = [
        center for center, members in subrow_clusters
        if len(members) >= MIN_SUBROW_DOTS
    ]

    if not valid_subrow_ys:
        return "[No valid dot rows found]"

    text_row_groups = cluster_1d_members(valid_subrow_ys, gap=v_sp * 2.5)

    # -------------------------------------------------------------------
    # FIX 4: Recover 3 sub-row positions per text row
    # -------------------------------------------------------------------
    text_rows = []
    for center, subrow_list in text_row_groups:
        row_ys = find_3_subrows(subrow_list, dot_set, v_sp)
        text_rows.append(row_ys)

    if not text_rows:
        return "[Could not detect Braille rows]"

    lines = []
    for row_ys in text_rows:
        y_lo = row_ys[0] - v_sp * 0.8
        y_hi = row_ys[2] + v_sp * 0.8
        row_dots = [(dx, dy) for (dx, dy) in dot_set if y_lo <= dy <= y_hi]
        if not row_dots:
            lines.append('')
            continue

        col_xs = cluster_1d([p[0] for p in row_dots], gap=intra * 0.5)
        cells = pair_columns(col_xs, intra, tol=intra * 0.35)

        row_chars = []
        for ki, (lx, rx) in enumerate(cells):
            if ki > 0:
                if cells[ki][0] - cells[ki - 1][1] > WORD_GAP:
                    row_chars.append(dots_to_uni(()))  # word space
            active = []
            for di in range(3):
                ry = row_ys[di]
                if any_near(dot_set, lx, ry, rad): active.append(di + 1)
                if any_near(dot_set, rx, ry, rad): active.append(di + 4)
            row_chars.append(dots_to_uni(tuple(sorted(set(active)))))

        lines.append(''.join(row_chars))

    return unicode_to_english('\n'.join(lines))


def decode_pdf(pdf_bytes: bytes) -> list[str]:
    """Convert PDF bytes → list of decoded English strings, one per page."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = []
    for page_num, page in enumerate(doc, start=1):
        try:
            pix = page.get_pixmap(dpi=300)
            pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pil_image_to_english(pil_img)
        except Exception as e:
            text = f"[Error on page {page_num}: {str(e)}]"
        all_text.append(text)
    return all_text