import pandas as pd

encodings = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
for enc in encodings:
    try:
        df = pd.read_csv('data/grade10_dataset.csv', encoding=enc)
        print(f'[OK] Successfully loaded with encoding: {enc}')
        print(f'[OK] Shape: {df.shape}')
        print(f'[OK] First chapter: {df.iloc[0]["chapter"]}')
        break
    except Exception as e:
        print(f'[FAIL] {enc}: {type(e).__name__}')
