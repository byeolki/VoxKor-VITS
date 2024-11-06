import re
from text.symbols import symbols
from text.cleaner import korean_cleaners

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def text_to_sequence(text):
  sequence = []
  clean_text = re.sub(r'\[KO\](.*?)\[KO\]', lambda x: korean_cleaners(x.group(1))+' ', text)    
  for symbol in clean_text:
    if symbol not in _symbol_to_id.keys():
      continue
    symbol_id = _symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence

def cleaned_text_to_sequence(cleaned_text):
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text if symbol in _symbol_to_id.keys()]
  return sequence