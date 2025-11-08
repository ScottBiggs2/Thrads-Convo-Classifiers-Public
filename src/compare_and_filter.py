#!/usr/bin/env python3
"""
Compare and Filter Labels from Gemini and GPT-4o

This script compares the labels from two different teacher models (Gemini and GPT-4o),
calculates the agreement rate, generates a confusion matrix, and creates a new
distillation-ready JSON file containing only the samples where the models agree.
"""

import json
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json(filepath: str):
    """Load a JSON or JSONL file."""
    logger.info(f"📂 Loading: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        if filepath.endswith(".jsonl"):
            return [json.loads(line) for line in f]
        else:
            return json.load(f)

def main():
    """Main function to compare and filter labels."""
    
    # File paths
    gemini_labels_path = "data/gemini_2.5_flash_labelled.json"
    gpt4o_labels_path = "data/distillation_data_gpt4o_mini.json"
    output_path = "data/agreed_distillation_data.json"
    confusion_matrix_path = "data/agreement_confusion_matrix.png"

    # Load the data
    gemini_data = load_json(gemini_labels_path)
    gpt4o_data = load_json(gpt4o_labels_path)

    # Create a dictionary for each model's labels, keyed by chat_id
    if isinstance(gemini_data, dict):
        gemini_labels = {item['chat_id']: item['teacher_prediction'] for item in gemini_data.get('distillation_ready', [])}
    else:
        gemini_labels = {item['chat_id']: item['intent'] for item in gemini_data}

    gpt4o_labels = {item['chat_id']: item['teacher_prediction'] for item in gpt4o_data.get('distillation_ready', [])}
    gpt4o_soft_labels = {item['chat_id']: item['soft_labels'] for item in gpt4o_data.get('distillation_ready', [])}
    gpt4o_text = {item['chat_id']: item['text'] for item in gpt4o_data.get('distillation_ready', [])}
    class_order = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M"
      ]
    #gpt4o_data.get('metadata', {}).get('class_order', [])

    # Debugging logs
    logger.info(f"Found {len(gemini_labels)} labels in {gemini_labels_path}")
    logger.info(f"Example chat_ids from Gemini: {list(gemini_labels.keys())[:5]}")
    logger.info(f"Found {len(gpt4o_labels)} labels in {gpt4o_labels_path}")
    logger.info(f"Example chat_ids from GPT-4o: {list(gpt4o_labels.keys())[:5]}")

    # Find common chat_ids
    common_chat_ids = set(gemini_labels.keys()) & set(gpt4o_labels.keys())
    logger.info(f"Found {len(common_chat_ids)} common chat_ids between the two files.")

    # Compare labels and filter for agreement
    agreed_samples = []
    y_true = []
    y_pred = []
    agreement_count = 0

    for chat_id in common_chat_ids:
        gemini_label = gemini_labels[chat_id]
        gpt4o_label = gpt4o_labels[chat_id]
        
        y_true.append(gemini_label)
        y_pred.append(gpt4o_label)

        if gemini_label == gpt4o_label:
            agreement_count += 1
            agreed_samples.append({
                'chat_id': chat_id,
                'text': gpt4o_text[chat_id],
                'hard_label': gemini_label, # or gpt4o_label, they are the same
                'soft_labels': gpt4o_soft_labels[chat_id],
                'class_order': class_order
            })

    # Calculate and log agreement rate
    agreement_rate = agreement_count / len(common_chat_ids) if common_chat_ids else 0
    logger.info(f"Agreement rate: {agreement_rate:.2%}")

    # Generate and save confusion matrix
    if common_chat_ids:
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Gemini vs. GPT-4o Label Agreement\nAgreement Rate: {agreement_rate:.2%}')
        plt.xlabel('GPT-4o Label')
        plt.ylabel('Gemini Label')
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Confusion matrix saved to {confusion_matrix_path}")
        plt.close()

    # Save the agreed samples to a new JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"distillation_ready": agreed_samples, "metadata": {"class_order": class_order}}, f, indent=2)
    
    logger.info(f"💾 Saved {len(agreed_samples)} agreed samples to {output_path}")

if __name__ == "__main__":
    main()
