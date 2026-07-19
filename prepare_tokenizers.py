"""
Regenerates sp_src.model / sp_tgt.model exactly as the notebook did in
cells 3-4 (download+clean) and cell 7 (SentencePiece training).
This part of the pipeline uses no randomness, so it reproduces the
original tokenizer files deterministically.
"""
import os
import sentencepiece as spm

SRC_FILE = "data/Urdu.txt"
TGT_FILE = "data/English.txt"
OUT_DIR = "models"

MAX_EN_LEN = 60
MIN_EN_LEN = 2

SRC_SP_VOCAB = 8000
TGT_SP_VOCAB = 8000


def read_parallel_corpus(src_path, tgt_path):
    with open(src_path, "r", encoding="utf-8") as f_src, \
         open(tgt_path, "r", encoding="utf-8") as f_tgt:
        src_lines = [l.strip() for l in f_src]
        tgt_lines = [l.strip() for l in f_tgt]
    assert len(src_lines) == len(tgt_lines)
    pairs = [(s, t) for s, t in zip(src_lines, tgt_lines) if s and t]
    src_clean, tgt_clean = zip(*pairs) if pairs else ([], [])
    return list(src_clean), list(tgt_clean)


def clean(src_sentences, tgt_sentences):
    clean_src, clean_tgt = [], []
    seen = set()
    for s, t in zip(src_sentences, tgt_sentences):
        s, t = s.strip(), t.strip()
        if not s or not t:
            continue
        len_t = len(t.split())
        if len_t > MAX_EN_LEN or len_t < MIN_EN_LEN:
            continue
        pair = (s, t)
        if pair in seen:
            continue
        seen.add(pair)
        clean_src.append(s)
        clean_tgt.append(t)
    return clean_src, clean_tgt


def main():
    assert os.path.exists(SRC_FILE) and os.path.exists(TGT_FILE), \
        f"{SRC_FILE} / {TGT_FILE} not found — download them first."
    os.makedirs(OUT_DIR, exist_ok=True)

    src_sentences, tgt_sentences = read_parallel_corpus(SRC_FILE, TGT_FILE)
    print(f"Raw pairs: {len(src_sentences)}")

    src_sentences, tgt_sentences = clean(src_sentences, tgt_sentences)
    print(f"Clean pairs: {len(src_sentences)}")

    with open("src_corpus.txt", "w", encoding="utf-8") as f:
        for s in src_sentences:
            f.write(s + "\n")
    with open("tgt_corpus.txt", "w", encoding="utf-8") as f:
        for t in tgt_sentences:
            f.write(t + "\n")

    print("Training source SentencePiece model...")
    spm.SentencePieceTrainer.Train(
        input="src_corpus.txt",
        model_prefix=f"{OUT_DIR}/sp_src",
        vocab_size=SRC_SP_VOCAB,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )

    print("Training target SentencePiece model...")
    spm.SentencePieceTrainer.Train(
        input="tgt_corpus.txt",
        model_prefix=f"{OUT_DIR}/sp_tgt",
        vocab_size=TGT_SP_VOCAB,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )

    sp_src = spm.SentencePieceProcessor()
    sp_src.load(f"{OUT_DIR}/sp_src.model")
    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.load(f"{OUT_DIR}/sp_tgt.model")

    print("Source vocab size:", sp_src.vocab_size())
    print("Target vocab size:", sp_tgt.vocab_size())
    print(f"Done. {OUT_DIR}/sp_src.model / {OUT_DIR}/sp_tgt.model written.")


if __name__ == "__main__":
    main()
