#!/usr/bin/env python3
"""
Extract conversation-level sequences from records.json.

- Converts each chat into a single "sequence" string by concatenating messages.
- Marks speaker boundaries with [USER] and [ASSISTANT] tokens.
- Omits 'context' and 'query' fields (do NOT include them).
- Outputs data/raw_sequences.json with items like:
  { "chat_id": "...", "sequence": " [USER] ... [ASSISTANT] ...", "createdAt": "..." }
"""

import os
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

INPUT_PATH = os.getenv("RECORDS_PATH", "data/records.json")
OUTPUT_PATH = os.getenv("RAW_SEQUENCES_PATH", "data/raw_sequences.json")
MAX_TURNS = int(os.getenv("MAX_TURNS_PER_SEQUENCE", 12))  # last N messages (both roles) by default

def build_sequence_from_chat(chat: dict, max_turns: int = MAX_TURNS):
    """
    Build a single string sequence from a chat's messages.
    Keeps the last `max_turns` messages (chronological).
    Prepends role markers [USER] and [ASSISTANT].
    """
    msgs = chat.get("messages", [])
    if not msgs:
        return None

    # Sort messages by createdAt if present (just in case)
    try:
        msgs_sorted = sorted(msgs, key=lambda m: m.get("createdAt", ""))
    except Exception:
        msgs_sorted = msgs

    # Keep last max_turns messages
    msgs_slice = msgs_sorted[-max_turns:]

    parts = []
    for m in msgs_slice:
        role = m.get("role", "").lower()
        content = m.get("content", "")
        if not content or not isinstance(content, str):
            continue
        content = content.strip()
        if role == "user":
            parts.append("[USER] " + content)
        elif role == "assistant":
            parts.append("[ASSISTANT] " + content)
        else:
            # Unknown role: include raw
            parts.append("[OTHER] " + content)

    if not parts:
        return None

    sequence_text = " \n".join(parts)
    return {
        "chat_id": chat.get("chat_id"),
        "sequence": sequence_text,
        "num_turns": len(parts),
        "source": "records.json"
    }

def extract_sequences(input_path=INPUT_PATH, output_path=OUTPUT_PATH, max_turns=MAX_TURNS):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # handle dict or list
    if isinstance(data, dict):
        chats = list(data.values())
    else:
        chats = data

    sequences = []
    for chat in tqdm(chats, desc="Extracting chat sequences"):
        seq = build_sequence_from_chat(chat, max_turns=max_turns)
        if seq:
            sequences.append(seq)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(sequences, fout, indent=2, ensure_ascii=False)

    print(f"✅ Extracted {len(sequences)} sequences -> {output_path}")

if __name__ == "__main__":
    extract_sequences()
