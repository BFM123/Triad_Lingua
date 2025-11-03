from transformers import MarianMTModel, MarianTokenizer
from utils import load_model_and_tokenizer

# Load once
ny_en_model, ny_en_tokenizer = load_model_and_tokenizer("Helsinki-NLP/opus-mt-ny-en")
en_hi_model, en_hi_tokenizer = load_model_and_tokenizer("Helsinki-NLP/opus-mt-en-hi")

def translate_chichewa_to_hindi(text: str, return_intermediate: bool = False):
    # Chichewa → English
    inputs = ny_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = ny_en_model.generate(**inputs)
    en_text = ny_en_tokenizer.decode(translated[0], skip_special_tokens=True)

    # English → Hindi
    inputs_hi = en_hi_tokenizer(en_text, return_tensors="pt", padding=True, truncation=True)
    translated_hi = en_hi_model.generate(**inputs_hi)
    hi_text = en_hi_tokenizer.decode(translated_hi[0], skip_special_tokens=True)

    if return_intermediate:
        return hi_text, en_text
    else:
        return hi_text
