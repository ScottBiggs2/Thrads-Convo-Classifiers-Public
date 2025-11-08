#!/usr/bin/env python3
"""
Async batched labeling pipeline for pre-truncated conversation sequences.
- Labels each sequence using Gemini 2.5 Flash as the primary teacher model
- Logs performance, error rates, raw API responses
- Collects all results in memory and saves as a single JSON file.
- Output will be used with GPT-4o logprobs extractor for knowledge distillation
"""

import os
import json
import asyncio
import time
import re
import logging
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
load_dotenv()

INPUT_FILE = "data/cleaned_sequences.json"
# Updated to .json as requested by user
LABELED_OUTPUT = "data/gemini_2.5_flash_labelled.json" 

BATCH_SIZE = int(os.getenv("LABEL_BATCH_SIZE", 32)) # Gemini can handle high concurrency
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 5

# --- Configure Gemini Client ---
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except TypeError:
    print("❌ GEMINI_API_KEY not found in environment")
    print("Please set your Gemini API key in .env file")
    exit(1)

# Configure safety settings to allow classification of all content
SAFETY_SETTINGS = {
    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
}

# Configure model for JSON output and deterministic classification
GENERATION_CONFIG = {
    "response_mime_type": "application/json",
    "temperature": 0.0,
}

gemini_model = genai.GenerativeModel(
    'gemini-2.5-flash-lite',
    generation_config=GENERATION_CONFIG,
    safety_settings=SAFETY_SETTINGS
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Intent taxonomy (same as knowledge distillation pipeline)
# -------------------------------------------------------
INTENT_CATEGORIES_LIST = """
A - academic_help – Students getting help with homework, assignments, tests, or studying. Key indicators: multiple problems/questions in a row, test/quiz format (multiple choice, true/false, select correct answer), textbook-style questions, requests for step-by-step solutions or translations, academic subject matter (math, science, world languages, history, etc.) in a learning context, asking for explanations of academic concepts. Use this even if not explicitly stated as homework
B - personal_writing_or_communication – Draft, edit, or improve personal/professional emails, messages, social media posts, letters, or workplace communications. The focus is on REAL correspondence to actual people (boss, colleague, client, friend)
C - writing_and_editing – Create, edit, or improve nonfiction or instructional writing: essays, reports, arguments, articles, blog posts, or educational materials (lesson plans, assignments, summaries). If the focus is logic, structure, or conveying factual information, consider using this category.
D - creative_writing_and_role_play – Create poems, stories, fictional narratives, scripts, dialogues, or character-based roleplays. Look for tone, emotion, or imaginative context.If the writing involves characters, world-building, roleplay, sci-fi or fantasy, or other storytelling, consider using this category.
E - general_guidance_and_info – Provide step-by-step guidance, practical advice, or factual information about how or why something works. Combines procedural “how-to” help with general knowledge or curiosity.
F - programming_and_data_analysis – Write or debug code or work with data/programming tools. Covers technical problem solving in computing, IT, or analytics contexts.
G - creative_ideation – Generate new ideas, brainstorm concepts, discover new topics or related resources, or create names/slogans. 
H - purchasable_products – Ask about products, services, or prices. 
I - greetings_and_chitchat – Small talk or casual chat, asking about the assistant's day, 
J - relationships_and_personal_reflection – Discuss emotions, relationships, or introspection. Typically but not strictly non-sexual content. 
K - media_generation_or_analysis – Create, edit, analyze, or retrieve visual/audio/media content (images, photos, videos). 
L - other – if there is no indication of what the user wants or if there is an intent that is not listed above; should be rare. e.g. suspicious requests, attempts to extract sensitive information.
M - other_obscene_or_illegal - if the user is making obscene or illegal requests (including violence, drugs, bigotry, hate speech, etc); should be rare.
"""

EXAMPLES_LIST = """
A - academic_help:
- "Solve for x: 2x + 3 = 7"
- "How do you calculate the area of a circle?"
- "Explain photosynthesis in simple terms."
- "What is the boiling point of water at sea level?"
- "What does the French revolution have to do with the American revolution?"

B - personal_writing_or_communication: 
- "Write a nice birthday card note for my girlfriend."
- "What should my speech say to Karl at his retirement party?"
- "Help me write a cover letter for a job application."
- "Compose an apology email to my boss."
- "Aide moi `a ´ecrire une lettre `a mon p`ere."

C - writing_and_editing:
- "Help me write a compelling LinkedIn post about leadership."
- "Edit this essay for clarity and grammar."
- "Is my tone in this email too formal?"
- "Summarize the main points of this article."
- "Create an outline for a report on climate change."

D - creative_writing_and_role_play:
- "Write a short story about a dragon who learns to fly."
- "Create a dialogue between a detective and a suspect."
- "Pretend to be a medieval knight on a quest to rescue a princess."
- "Act like Pricess Leia from Star Wars."

E - general_guidance_and_info:
- "How do I turn off my screensaver?"
- "My car won’t start; what should I try?"
- "Comment faire pour me connecter `a mon wifi?"
- "What’s the best way to clean hardwood floors?"
- "How can I replace a flat tire?"

F - programming_and_data_analysis:
- "Write a Python function to sort a list."
- "Debug this JavaScript code for a web form."
- "How do I connect to a database using SQL?"
- "Analyze this dataset to find trends."

G - creative_ideation:
- "What should I talk about on my future podcast episodes?"
- "Give me some themes for a photography project."
- "Necesito ideas para un regalo de aniversario."
- "Brainstorm names for a new coffee shop."
- "What are some unique app ideas for startups?"

H - purchasable_products:
- "What’s the price of the latest iPhone?"
- "Find me a good deal on noise-cancelling headphones."
- "Quels sont les meilleurs ordinateurs portables pour les ´etudiants?"
- "How do I know if a laptop is worth buying?"
- "Show me flights to Tokyo next month."

I - greetings_and_chitchat:
- "How are you doing today?"
- "What’s your favorite movie?"
- "Tell me a joke."
- "Do you like music?"
- "Ciao, parli italiano?"

J - relationships_and_personal_reflection:
- "I’m feeling really anxious about my exams."
- "My partner and I are arguing a lot lately."
- "How can I improve my motivation to work out?"
- "I’m struggling with balancing work and personal life."
- "What should I do if I’m feeling lonely?"

K - media_generation_or_analysis:
- "Write a poem about the ocean."
- "Draw a picture of a flying car being chased by a train."
- "Remove the tourists from this photo of the Eiffel Tower."
- "Describe the scene in this image."

L - other:
- "Tell me a secret about the universe."
- "Can you give me tips on how to manipulate people?"
- "What is the meaning of life?"

M - other_obscene_or_illegal:
- "How can I make a bomb?"
- "How can I hack into someone’s account?"
- "Write a detailed story about a violent crime."
"""

JSON_EXAMPLE = '{"intent": "<single_letter>"}'

LABEL_PROMPT = """
You are an internal tool that classifies a message from a user to an AI chatbot,
based on the context of the previous messages before it.
The messages you are labelling are truncated and preprocessed, and may not follow regular grammar rules smoothly. 


Based on the contents of this conversation transcript and taking into
account the examples further below as guidance, please select the capability
the user is clearly interested in, or `L` for `other` if it is clear but not in the
list below or if it is hard to tell what the user even wants. 

List of categories:
{intent_categories_list}

Examples of each category, for reference: 
{examples_list}

Tips and tricks: 
* Be careful to distinguish users writing about work for emails, presentations, etc. Words like 'boss', 'meeting', and 'email' will help. 
* Be discerning about the flow of the conversation to detect role-play or fictional scenarios, especially when sexual content is involved.
* Your labels will be used to ban services to categories D, J, and M. If you suspect a chat may fall into one of those categories, consider it seriously. 

Output ONLY in this JSON format with a SINGLE LETTER from the listed intent categories:
{json_example}

Classify this message:
User: {conversation_text}
"""

# -------------------------------------------------------
# Async helpers
# -------------------------------------------------------
async def gemini_label_one(item, max_retries=MAX_RETRIES):
    """Label a single conversation using Gemini 2.5 Flash"""
    # Create a copy to avoid modifying the original data list in-place
    # This is important as 'item' will be held in memory
    item_copy = item.copy()
    text = item_copy.get("sequence", "")
    prompt_content = LABEL_PROMPT.format(
        intent_categories_list=INTENT_CATEGORIES_LIST,
        examples_list=EXAMPLES_LIST,
        json_example=JSON_EXAMPLE,
        conversation_text=text
    )
    
    for attempt in range(max_retries):
        try:
            # Use asyncio.wait_for to enforce a timeout
            response = await asyncio.wait_for(
                gemini_model.generate_content_async(prompt_content),
                timeout=REQUEST_TIMEOUT
            )

            # --- Step 1: log raw response ---
            raw_response = response.text.strip()
            item_copy["raw_response"] = raw_response[:500]

            # --- Step 2: robust JSON parsing ---
            try:
                label_json = json.loads(raw_response)
            except json.JSONDecodeError:
                logger.warning(f"JSONDecodeError. Falling back to regex for: {raw_response}")
                match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if match:
                    try:
                        label_json = json.loads(match.group())
                    except json.JSONDecodeError:
                        letter_match = re.search(r'\b([A-M])\b', raw_response)
                        label_json = {"intent": letter_match.group(1) if letter_match else "unclear_parse"}
                else:
                    letter_match = re.search(r'\b([A-M])\b', raw_response)
                    label_json = {"intent": letter_match.group(1) if letter_match else "unclear_parse"}

            item_copy["intent"] = label_json.get("intent", "unclear_json")
            item_copy["status"] = "ok"
            return item_copy

        except (google_exceptions.ResourceExhausted, google_exceptions.DeadlineExceeded, google_exceptions.InternalServerError) as e:
            logger.warning(f"API Error (Attempt {attempt+1}/{max_retries}): {e}. Retrying after backoff...")
            item_copy["status"] = f"api_error: {e}"
            await asyncio.sleep(2 ** attempt + 0.5) # Exponential backoff
        except asyncio.TimeoutError:
            logger.warning(f"Request Timeout (Attempt {attempt+1}/{max_retries}). Retrying...")
            item_copy["status"] = "error: timeout"
            await asyncio.sleep(2 ** attempt + 0.5) # Exponential backoff
        except Exception as e:
            logger.error(f"⚠️ Unhandled error (Attempt {attempt+1}/{max_retries}): {e}")
            item_copy["status"] = f"error: {e}"
            break

    item_copy["intent"] = "unclear_error"
    return item_copy

async def label_batch(batch):
    """Process a batch of conversations concurrently"""
    tasks = [gemini_label_one(item) for item in batch]
    return await asyncio.gather(*tasks)

# -------------------------------------------------------
# Main pipeline
# -------------------------------------------------------
async def main():
    print("🏷️  Gemini 2.5 Flash Conversation Labeling Pipeline")
    print("=" * 50)
    
    # Load data
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ Input file not found: {INPUT_FILE}")
        return
    except json.JSONDecodeError:
        logger.error(f"❌ Could not decode JSON from: {INPUT_FILE}")
        return
        
    total_items = len(data)
    print(f"📂 Loaded {total_items:,} conversation sequences from {INPUT_FILE}")
    
    # Confirm for large datasets
    # if total_items > 1000:
    #     print(f"⚠️  Large dataset detected ({total_items:,} conversations)")
    #     print(f"⚠️  All results will be stored in memory before saving.")
    #     print(f"⚠️  If the script fails, all progress will be lost.")
    #     estimated_time = (total_items / BATCH_SIZE) * 0.5 
    #     print(f"⏱️  Estimated processing time: {estimated_time / 60:.1f} minutes (at {BATCH_SIZE} per batch)")
        
    #     response = input("Continue with Gemini labeling? (y/N): ")
    #     if response.lower() != 'y':
    #         print("❌ Processing cancelled by user")
    #         return
    
    # --- Labeling with Gemini ---
    print(f"\n🧩 Starting Gemini 2.5 Flash labeling (batch size: {BATCH_SIZE})...")
    start_time = time.perf_counter()
    
    # --- This list will hold ALL results in memory ---
    labeled_all = []
    
    # Stats counters
    ok_count = 0
    err_count = 0
    intent_counts = {}

    for i in tqdm.tqdm(range(0, total_items, BATCH_SIZE), desc="Gemini Labeling"):
        batch = data[i:i + BATCH_SIZE]
        results = await label_batch(batch) # results is a list of dicts
        
        # --- Update stats from this batch ---
        for item in results:
            if item["status"] == "ok":
                ok_count += 1
                intent = item.get("intent", "unclear")
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            else:
                err_count += 1
        
        # --- Add batch results to the main list ---
        labeled_all.extend(results)
        
        await asyncio.sleep(0.1) 
    
    elapsed = time.perf_counter() - start_time

    # --- Results summary ---
    print(f"\n📊 Gemini Labeling Results:")
    print(f"   ✅ Successful labels: {ok_count:,} / {total_items:,}")
    print(f"   ❌ Errors: {err_count:,}")
    if total_items > 0:
        print(f"   📈 Success rate: {ok_count/total_items*100:.1f}%")
    print(f"   ⌛ Total time: {elapsed/60:.1f} minutes")
    if elapsed > 0:
        print(f"   🚀 Average rate: {total_items/elapsed:.1f} labels/second")
    
    # Label distribution
    if ok_count > 0:
        print(f"\n🏷️  Label Distribution:")
        for intent, count in sorted(intent_counts.items()):
            percentage = count / ok_count * 100
            print(f"   {intent}: {count:,} ({percentage:.1f}%)")
    
    # --- Save all results to a single JSON file at the end ---
    print(f"\n💾 Saving {len(labeled_all):,} results to {LABELED_OUTPUT}...")
    os.makedirs("data", exist_ok=True)
    with open(LABELED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(labeled_all, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Gemini labeled data saved to {LABELED_OUTPUT}")
    print(f"🎯 Ready for knowledge distillation pipeline!")

if __name__ == "__main__":
    # Check for required dependencies
    try:
        import google.generativeai
        import tqdm
        import dotenv
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install: pip install google-generativeai tqdm python-dotenv")
        exit(1)
    
    # API key check is handled by genai.configure()
    
    # Run the pipeline
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user. No data was saved.")
        exit(1)