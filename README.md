# Urdu → English Transformer

A Transformer (encoder-decoder) machine translation model built from scratch
in PyTorch, trained to translate **Urdu → English**. No pretrained weights,
no fine-tuning — the full pipeline (tokenizer, architecture, training loop)
is implemented and trained from zero in [English_to_Urdu_Transformer.ipynb](English_to_Urdu_Transformer.ipynb).

## Results

Evaluated on a held-out test set of 9,075 sentence pairs:

| Metric | Score |
|--------|-------|
| BLEU   | 31.10 |
| chrF   | 55.92 |
| TER    | 55.12 |

The model handles general/news-style sentences well but can struggle with
proper nouns, rare vocabulary, and long or grammatically complex sentences —
expected for a ~52M-parameter model trained from scratch on ~91K sentence
pairs (small by MT standards).

## Repo layout

```
English_to_Urdu_Transformer.ipynb   Full pipeline: data prep, tokenizer
                                     training, model, training loop, eval
translate.py                        Load the trained model and translate
                                     Urdu sentences from the command line
prepare_tokenizers.py               Regenerates the SentencePiece tokenizers
                                     from data/ (deterministic, see below)
data/
  Urdu.txt, English.txt             Parallel corpus (100K sentence pairs)
models/
  final_model.pt                    Trained model weights + config (git-lfs)
  sp_src.model, sp_tgt.model        SentencePiece BPE tokenizers (Urdu, English)
deployment/hf_space/                Gradio app prepared for Hugging Face
                                     Spaces (not yet deployed)
```

## How it works

1. **Data**: 100K Urdu-English sentence pairs, cleaned and deduplicated down
   to ~91K, split 80/10/10 into train/val/test.
2. **Tokenization**: two separate SentencePiece BPE tokenizers (8,000 tokens
   each) — one for Urdu, one for English.
3. **Model**: a 6-encoder/6-decoder-layer pre-norm Transformer, 512-dim
   embeddings, 8 heads, weight-tied output projection (~52M parameters).
4. **Training**: 15 epochs, Adam, label smoothing, mixed precision, early
   stopping on validation loss.
5. **Decoding**: beam search (beam size 4) at inference time.

## Running it yourself

The trained weights (`models/final_model.pt`) only make sense paired with
the exact tokenizers they were trained against. Those tokenizers are
included in `models/`, so you can translate directly:

```bash
pip install torch sentencepiece
python translate.py
```

This drops you into a prompt — type an Urdu sentence, get an English
translation back.

### Regenerating the tokenizers from scratch

If you ever need to rebuild `sp_src.model` / `sp_tgt.model` (e.g. you only
have `final_model.pt` and not the `models/` tokenizer files), they're
reproducible deterministically from `data/Urdu.txt` + `data/English.txt`:

```bash
python prepare_tokenizers.py
```

This reruns the same cleaning + BPE training used in the notebook (no
randomness involved in this part of the pipeline), so it reproduces
byte-identical tokenizer files.

## Deployment

`deployment/hf_space/` contains a ready-to-go Gradio app for Hugging Face
Spaces (not yet deployed). See that folder's README for details.

## Acknowledgments

The parallel Urdu-English corpus (`data/Urdu.txt`, `data/English.txt`) comes
from [Kheem-Dh/Urdu-to-English-Machine-Translation-using-Seq2Seq-Transformers-Variant-Model](https://github.com/Kheem-Dh/Urdu-to-English-Machine-Translation-using-Seq2Seq-Transformers-Variant-Model).
Only the dataset is reused — the model, tokenizer, training pipeline, and
code in this repo are an independent implementation.
