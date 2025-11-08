# Message Cleaning: 

Beginning with a ```records.json``` in the data folder. Example of structure: 
```json    
    "chat_id": "sample",
    "context": "sample",
    "query": "sample",
    "messages": [
      {
        "id": "sample",
        "createdAt": "sample",
        "role": "user",
        "content": "stuff"
      },
      {
        "id": "sample",
        "createdAt": "sample",
        "role": "assistant",
        "content": "stuff"
      },
      {
        "id": "sample",
        "createdAt": "sample",
        "role": "user",
        "content": "stuff"
      },
      {
        "id": "sample",
        "createdAt": "sample",
        "role": "assistant",
        "content": "stuff"
      }
    ],
    "formatted_chat": "User: stuff"
```

*** TODO: Go make sure that long conversation histories are properly dealt with ***

```bash 
python src/extract_sequences.py
```

You will need to change the `max_tokens` arg here to 512 or 1024 depending on your target token window. Currently defaults to 1024.
```bash 
python src/preprocess_sequences.py
```
Very simple extraction script. 

# Labelling
Get Gemini 2.5 Flash and GPT 4o mini lables (can be run in parallel with a simple nohup .sh script):
```bash
python src/get_gemini_flash_labels.py
```

```bash 
python src/get_gpt_4o_mini_labels.py
```

To verify labels with gemini flash run (will save a NEW json of only AGREED on samples). 

```bash 
python src/compare_and_filter.py
```
Provides some reporting/visibility on agreement rates and class breakdowns in 'true' data. 

# Then... 

For training distilBERTs and BERTs, run: 
```bash
python src/train_distilBERT.py
```
You can modify the base model you train by changing the `model_name` string in the config. To use soft labels, modify `alpha` in your .env and the [MORE]


# Export as ONNX, in case autosave fails:

**DistilBERT**
```bash
python src/export_to_onnx.py --model-path models/distilbert_512_intent_classifier --output-path models/distilbert_512_intent_classifier_onnx --max-length 512 --test
```

# Evaluation: 
Supposing you have `records.json` in `data/`, 

First, run (you need to manually change `num_samples` for now): 
```bash
python eval/preprocess_data.py
```

Then, 
```bash
python eval/run_evaluation.py 
```

