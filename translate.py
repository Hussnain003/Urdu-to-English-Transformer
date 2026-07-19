"""
Interactive Urdu -> English translation using the trained model
(final_model.pt) and the regenerated SentencePiece tokenizers
(sp_src.model / sp_tgt.model). Mirrors cells 11, 12, 17 of the notebook.
"""
import io
import math
import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MAX_SEQ_LEN = 60

SPECIAL_TOKENS = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
PAD_IDX = SPECIAL_TOKENS["<pad>"]
SOS_IDX = SPECIAL_TOKENS["<sos>"]
EOS_IDX = SPECIAL_TOKENS["<eos>"]
UNK_IDX = SPECIAL_TOKENS["<unk>"]
SRC_OFFSET = len(SPECIAL_TOKENS)
TGT_OFFSET = len(SPECIAL_TOKENS)

MODEL_DIR = "models"

sp_src = spm.SentencePieceProcessor()
sp_src.load(f"{MODEL_DIR}/sp_src.model")
sp_tgt = spm.SentencePieceProcessor()
sp_tgt.load(f"{MODEL_DIR}/sp_tgt.model")


def encode_src(text: str, max_len: int) -> List[int]:
    sp_ids = sp_src.EncodeAsIds(text)
    sp_ids = sp_ids[: max_len - 2]
    return [SOS_IDX] + [i + SRC_OFFSET for i in sp_ids] + [EOS_IDX]


def decode_tgt_ids(ids: List[int]) -> str:
    pieces = []
    for idx in ids:
        if idx in (PAD_IDX, SOS_IDX, EOS_IDX):
            continue
        if idx < TGT_OFFSET:
            continue
        sp_id = idx - TGT_OFFSET
        if sp_id < sp_tgt.vocab_size():
            pieces.append(sp_tgt.IdToPiece(sp_id))
    return sp_tgt.DecodePieces(pieces)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ImprovedSeq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                 src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.output_norm = nn.LayerNorm(emb_size)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.generator.weight = self.tgt_tok_emb.weight


def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz), device=device) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
    return mask


def create_padding_mask(seq, pad_idx):
    return seq == pad_idx


def beam_search_decode(model, src, beam_size=4, max_len=MAX_SEQ_LEN, length_penalty=0.7):
    model.eval()
    src = src.to(device)
    src_seq_len = src.size(1)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    src_padding_mask = create_padding_mask(src, PAD_IDX)

    with torch.no_grad():
        src_emb = model.positional_encoding(model.src_tok_emb(src) * math.sqrt(model.emb_size))
        memory = model.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

    beams = [(torch.tensor([SOS_IDX], device=device, dtype=torch.long), 0.0)]

    for _ in range(max_len - 1):
        new_beams = []
        for seq, log_prob in beams:
            if seq[-1].item() == EOS_IDX:
                new_beams.append((seq, log_prob))
                continue

            ys = seq.unsqueeze(0)
            tgt_len = ys.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_len)
            tgt_padding_mask = create_padding_mask(ys, PAD_IDX)

            with torch.no_grad():
                tgt_emb = model.positional_encoding(model.tgt_tok_emb(ys) * math.sqrt(model.emb_size))
                out = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                     memory_key_padding_mask=src_padding_mask,
                                     tgt_key_padding_mask=tgt_padding_mask)
                out_step = model.generator(model.output_norm(out[:, -1, :]))
                log_probs = F.log_softmax(out_step, dim=-1).squeeze(0)
                log_probs[PAD_IDX] = -1e9
                log_probs[SOS_IDX] = -1e9
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)

            for k in range(beam_size):
                next_id = topk_ids[k].item()
                next_log_prob = log_prob + topk_log_probs[k].item()
                new_seq = torch.cat([seq, torch.tensor([next_id], device=device, dtype=torch.long)], dim=0)
                new_beams.append((new_seq, next_log_prob))

        def score_fn(b):
            seq, logp = b
            return logp / (len(seq) ** length_penalty)

        new_beams.sort(key=score_fn, reverse=True)
        beams = new_beams[:beam_size]

        if all(b[0][-1].item() == EOS_IDX for b in beams):
            break

    best_seq, _ = beams[0]
    return best_seq.tolist()


def translate_sentence(model, src_text: str, beam_size: int = 4) -> str:
    ids = encode_src(src_text, MAX_SEQ_LEN)
    src_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    pred_ids = beam_search_decode(model, src_tensor, beam_size=beam_size)
    return decode_tgt_ids(pred_ids)


def load_model(checkpoint_path=f"{MODEL_DIR}/final_model.pt"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = ImprovedSeq2SeqTransformer(
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        emb_size=cfg["embed_dim"],
        nhead=cfg["n_head"],
        src_vocab_size=cfg["src_vocab_size"],
        tgt_vocab_size=cfg["tgt_vocab_size"],
        dim_feedforward=cfg["ffn_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Loaded model. Test results from training run:", ckpt.get("test_results"))
    return model


if __name__ == "__main__":
    # Force UTF-8 so Urdu input/output doesn't crash on Windows consoles
    # whose default codepage (e.g. cp1252) can't represent Urdu characters.
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    model = load_model()
    print("\nType an Urdu sentence to translate (or 'quit' to exit):\n")
    while True:
        try:
            text = input("UR> ").strip()
        except EOFError:
            break
        if not text or text.lower() in ("quit", "exit"):
            break
        print("EN>", translate_sentence(model, text))
