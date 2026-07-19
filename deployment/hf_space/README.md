---
title: Urdu to English Translator
emoji: 🌐
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Urdu → English Translator

A from-scratch Transformer (encoder-decoder, ~52M params) trained on ~91K
Urdu-English sentence pairs, using SentencePiece BPE tokenization (8000 tokens
per language).

Test set: BLEU 31.10, chrF 55.92, TER 55.12.

Note: this is a small model trained from scratch on a modest dataset — it
handles general/news-style sentences well but can struggle with proper nouns,
rare vocabulary, and long or technical sentences.

## Deploying this Space

This folder intentionally does not include the model/tokenizer files (to
avoid duplicating ~210MB in the main repo). Before pushing this folder to a
Hugging Face Space, copy these in from `../../models/`:

```
final_model.pt
sp_src.model
sp_tgt.model
```

`app.py` expects them alongside itself in this same directory.
