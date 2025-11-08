#!/usr/bin/env python3
"""
Preprocess chat sequences:
- Split by [USER] / [ASSISTANT] markers
- Take last 3 turns (6 messages)
- For each message, keep ~170 tokens; if longer, keep first 85 + last 85 tokens
- Lowercase everything
- Save as JSON, preserving all chat_ids
"""
import json
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
MAX_TOKENS_PER_MSG = 170
HALF_TOKENS_PER_MSG = 85  # first + last 85 tokens if trimmed

def process_message(msg, tokenizer):
    """Trim a single message to ~170 tokens, or first/last 85 if longer."""
    tokens = tokenizer.encode(msg, add_special_tokens=False)
    if len(tokens) <= MAX_TOKENS_PER_MSG:
        return tokenizer.decode(tokens, skip_special_tokens=True)
    else:
        first_half = tokens[:HALF_TOKENS_PER_MSG]
        last_half = tokens[-HALF_TOKENS_PER_MSG:]
        return tokenizer.decode(first_half, skip_special_tokens=True) + " [...] " + tokenizer.decode(last_half, skip_special_tokens=True)

def process_conversation_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    processed = []
    for convo in data:
        chat_id = convo.get("chat_id", "")
        text = convo.get("sequence", "")
        num_turns = convo.get("num_turns", None)
        if not text or not isinstance(text, str):
            processed.append({"chat_id": chat_id, "sequence": "", "num_turns": num_turns})
            continue
        text = text.lower()
        # Split into turns by [USER] / [ASSISTANT]
        segments = []
        for seg in text.split("[user]"):
            sub_segs = seg.split("[assistant]")
            for s in sub_segs:
                s = s.strip()
                if s:
                    segments.append(s)
        # Take last 3 turns (6 messages)
        last_turns = segments[-6:] if len(segments) >= 6 else segments
        # Rebuild sequence with markers, trimming each message
        sequence_text = ""
        for i, msg in enumerate(last_turns):
            role = "[USER]" if i % 2 == 0 else "[ASSISTANT]"
            trimmed_msg = process_message(msg, tokenizer)
            sequence_text += f"{role} {trimmed_msg} "
        processed.append({
            "chat_id": chat_id,
            "sequence": sequence_text,
            "num_turns": num_turns
        })
    # Save JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(processed)} sequences to {output_path}")

if __name__ == "__main__":
    input_file = "data/raw_sequences.json"
    output_file = "data/cleaned_sequences.json"
    process_conversation_file(input_file, output_file)
