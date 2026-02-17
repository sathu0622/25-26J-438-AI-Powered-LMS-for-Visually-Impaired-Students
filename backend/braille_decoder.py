import fitz  # PyMuPDF
import re

BRAILLE_ALPHA = {
    "⠁":"a","⠃":"b","⠉":"c","⠙":"d","⠑":"e",
    "⠋":"f","⠛":"g","⠓":"h","⠊":"i","⠚":"j",
    "⠅":"k","⠇":"l","⠍":"m","⠝":"n","⠕":"o",
    "⠏":"p","⠟":"q","⠗":"r","⠎":"s","⠞":"t",
    "⠥":"u","⠧":"v","⠺":"w","⠭":"x","⠽":"y","⠵":"z"
}

BRAILLE_NUMBERS = {
    "⠁":"1","⠃":"2","⠉":"3","⠙":"4","⠑":"5",
    "⠋":"6","⠛":"7","⠓":"8","⠊":"9","⠚":"0"
}

BRAILLE_PUNCT = {
    "⠂":",",
    "⠲":".",
    "⠦":"“",
    "⠴":"”",
    "⠖":"!",
    "⠶":"?",
    "⠤":"-",
    "⠐":":",
    "⠆":";",
    "⠀":" "
}

CAPITAL_SIGN = "⠠"
NUMBER_SIGN = "⠼"

# Extract Braille Unicode Text from PDF
def extract_braille_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    braille_text = ""

    for page in doc:
        braille_text += page.get_text()

    return braille_text

# Convert Braille Unicode → English
def braille_to_english(text: str) -> str:
    result = ""
    capitalize_next = False
    number_mode = False

    for ch in text:

        if ch == CAPITAL_SIGN:
            capitalize_next = True
            continue

        if ch == NUMBER_SIGN:
            number_mode = True
            continue

        if ch == "⠀":
            number_mode = False
            result += " "
            continue

        if number_mode and ch in BRAILLE_NUMBERS:
            result += BRAILLE_NUMBERS[ch]
            continue

        if ch in BRAILLE_ALPHA:
            letter = BRAILLE_ALPHA[ch]
            if capitalize_next:
                letter = letter.upper()
                capitalize_next = False
            result += letter
            continue

        if ch in BRAILLE_PUNCT:
            result += BRAILLE_PUNCT[ch]
            continue

        result += ch

    return result

# Fix Broken Line Words
def clean_wrapped_lines(text: str) -> str:
    return re.sub(r"(\w+)\n(\w+)", r"\1 \2", text)

# Split Question + Answer
def split_question_answer(text: str):
    text = text.strip()

    parts = re.split(r"\n\s*\n", text, maxsplit=1)

    if len(parts) == 2:
        question = clean_wrapped_lines(parts[0]).strip()
        answer = clean_wrapped_lines(parts[1]).strip()
        return question, answer

    return text, ""

# Main Function: PDF → Question + Answer
def decode_braille_pdf(pdf_path: str):

    braille_text = extract_braille_text(pdf_path)
    english_text = braille_to_english(braille_text)

    question, answer = split_question_answer(english_text)

    return {
        "full_text": english_text,
        "question": question,
        "answer": answer
    }
