import cv2
import numpy as np
import tensorflow as tf
from config import TFLITE_MODEL_PATH, BRAILLE_LABELS

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_cell(img_cell):
    img = cv2.resize(img_cell, (64, 64))
    img = img / 255.0
    input_data = np.expand_dims(img.astype(np.float32), axis=0)
    if input_details[0]['shape'][-1] == 3 and img.shape[-1] != 3:
        input_data = np.repeat(input_data, 3, axis=-1)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return BRAILLE_LABELS[np.argmax(output)]

def braille_image_to_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold: dots become white blobs
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find dot contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    dots = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Filter dot size
        if 5 < w < 25 and 5 < h < 25:
            dots.append((x, y, w, h))

    if len(dots) == 0:
        return ""

    # Sort dots left-to-right
    dots = sorted(dots, key=lambda d: d[0])

    # --- GROUP INTO BRAILLE CELLS ---
    cells = []
    current_cell = [dots[0]]

    for i in range(1, len(dots)):
        prev_x = dots[i - 1][0]
        curr_x = dots[i][0]

        # If gap is large → new character cell
        if curr_x - prev_x > 35:
            cells.append(current_cell)
            current_cell = []

        current_cell.append(dots[i])

    if current_cell:
        cells.append(current_cell)

    # --- PREDICT EACH CELL ---
    text = ""

    for cell in cells:
        x_vals = [d[0] for d in cell]
        y_vals = [d[1] for d in cell]
        w_vals = [d[2] for d in cell]
        h_vals = [d[3] for d in cell]

        x1 = min(x_vals)
        y1 = min(y_vals)
        x2 = max([x + w for x, w in zip(x_vals, w_vals)])
        y2 = max([y + h for y, h in zip(y_vals, h_vals)])

        cell_img = gray[y1:y2, x1:x2]

        if cell_img.size == 0:
            continue

        cell_img = cv2.cvtColor(cell_img, cv2.COLOR_GRAY2BGR)

        pred_char = predict_cell(cell_img)
        text += pred_char

    return text
