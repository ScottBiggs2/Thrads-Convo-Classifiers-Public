#!/usr/bin/env python3
"""
OpenAI GPT-4o-mini Single-Pass Labeling and Logprob Extraction
Fixed version with proper timeout and interrupt handling
"""

import json
import asyncio
import logging
import os
import math
import re
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import openai
import matplotlib.pyplot as plt
from collections import Counter

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ConversationData:
    """Data structure for conversation with labels and logits"""
    chat_id: str
    sequence: str
    sequence_truncated: str
    gpt4o_label: str
    openai_label: str
    openai_logprobs: List[float]
    openai_tokens: List[str]
    agreement: bool

def save_progress(results: List[ConversationData], temp_output_file: str):
    """Saves the current progress to a temporary file."""
    with open(temp_output_file, 'w', encoding='utf-8') as f:
        json.dump([res.__dict__ for res in results], f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Progress saved to {temp_output_file}")

class OpenAILabeler:
    """Single-pass labeling and logits extraction with OpenAI"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.AsyncOpenAI(api_key=api_key, max_retries=0)
        self.model = model
        self.valid_class_tokens = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'}
        self.all_valid_tokens_sorted = sorted(list(self.valid_class_tokens))
        logger.info(f"✅ OpenAI client initialized (retries disabled)")
        logger.info(f"🤖 Model: {model}")

    def create_classification_prompt(self, conversation_text: str) -> str:
        """Create classification prompt for OpenAI"""
        INTENT_CATEGORIES_LIST = """
        A - academic_help – Students getting help with homework, assignments, tests, or studying.
        B - personal_writing_or_communication – Draft, edit, or improve personal/professional emails, messages, social media posts, letters, or workplace communications.
        C - writing_and_editing – Create, edit, or improve nonfiction or instructional writing.
        D - creative_writing_and_role_play – Create poems, stories, fictional narratives, scripts, dialogues, or character-based roleplays.
        E - general_guidance_and_info – Provide step-by-step guidance, practical advice, or factual information.
        F - programming_and_data_analysis – Write or debug code or work with data/programming tools.
        G - creative_ideation – Generate new ideas, brainstorm concepts, or discover new topics.
        H - purchasable_products – Ask about products, services, or prices.
        I - greetings_and_chitchat – Small talk or casual chat.
        J - relationships_and_personal_reflection – Discuss emotions, relationships, or introspection.
        K - media_generation_or_analysis – Create, edit, analyze, or retrieve visual/audio/media content.
        L - other – if there is no indication of what the user wants or if there is an intent that is not listed above.
        M - other_obscene_or_illegal - if the user is making obscene or illegal requests.
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


        LABEL_PROMPT = f"""
        You are an internal tool that classifies a message from a user to an AI chatbot,
        based on the context of the previous messages before it. 
        The messages you are labelling are truncated and preprocessed, and may not follow regular grammar rules smoothly. 

        Based on the contents of this conversation transcript please select the capability
        the user is clearly interested in, or `L` for `other` if it is clear but not in the
        list below or if it is hard to tell what the user even wants. 

        List of categories:
        {INTENT_CATEGORIES_LIST}

        Examples of each category, for reference: 
        {EXAMPLES_LIST}

        Tips and tricks: 
        * Be careful to distinguish users writing about work for emails, presentations, etc. Words like 'boss', 'meeting', and 'email' will help. 
        * Be discerning about the flow of the conversation to detect role-play or fictional scenarios, especially when sexual content is involved.
        * Your labels will be used to ban services to categories D, J, and M. If you suspect a chat may fall into one of those categories, consider it seriously. 

        Output ONLY the single letter classification (A, B, C, D, E, F, G, H, I, J, K, L, M). No JSON, no explanation, just the letter.

        Classify this message:
        User: {conversation_text}

        Classification:
        """

        return LABEL_PROMPT

    def filter_and_softmax_logprobs(self, tokens: List[str], logprobs: List[float]) -> Dict[str, float]:
        """Filter logprobs to only include valid class tokens and apply softmax."""
        if not tokens or not logprobs:
            return {}
        
        token_logprob_map = {token.strip().upper(): logprob for token, logprob in zip(tokens, logprobs) if token.strip().upper() in self.valid_class_tokens}

        if not token_logprob_map:
            return {}

        # Apply softmax
        exp_log_probs = [math.exp(lp) for lp in token_logprob_map.values()]
        sum_exp_log_probs = sum(exp_log_probs)
        
        if sum_exp_log_probs > 0:
            probs = {token: math.exp(lp) / sum_exp_log_probs for token, lp in token_logprob_map.items()}
        else:
            probs = {token: 1.0 / len(token_logprob_map) for token in token_logprob_map}
        
        return probs

    async def process_with_realtime_api(self, conversations: List[Dict], max_concurrent: int = 5, progress_interval: int = 100, temp_output_file: str = "temp_results.json") -> List[ConversationData]:
        """Process with real-time API using concurrent batching with rate limiting"""

        logger.info(f"🔄 Using real-time API with {max_concurrent} concurrent requests...")
        
        await asyncio.sleep(2)

        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(conv, text):
            async with semaphore:
                return await self._process_single_realtime(conv, text)
        
        task_inputs = []
        for conv in conversations:
            text = conv.get('formatted_chat', conv.get('sequence_truncated', conv.get('sequence', '')))
            if text:
                task_inputs.append((conv, text))
        
        logger.info(f"📦 Processing {len(task_inputs)} requests...")
        
        results = []
        start_time = time.time()
        active_tasks = set()
        delay_between_starts = 1.0
        last_start_time = 0
        
        try:
            for idx, (conv, text) in enumerate(task_inputs):
                now = time.time()
                time_since_last_start = now - last_start_time
                if time_since_last_start < delay_between_starts:
                    await asyncio.sleep(delay_between_starts - time_since_last_start)
                
                task = asyncio.create_task(process_with_limit(conv, text))
                active_tasks.add(task)
                last_start_time = time.time()
                
                done_tasks = {t for t in active_tasks if t.done()}
                for done_task in done_tasks:
                    try:
                        result = await done_task
                        if result:
                            results.append(result)
                    except Exception:
                        pass
                active_tasks -= done_tasks
                
                if (idx + 1) % progress_interval == 0 or (idx + 1) == len(task_inputs):
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed if elapsed > 0 else 0
                    logger.info(f"   Progress: {idx + 1}/{len(task_inputs)} ({(idx + 1)/len(task_inputs)*100:.1f}%) | "
                               f"Collected: {len(results)} | Active: {len(active_tasks)} | Rate: {rate:.1f} req/s")
                    save_progress(results, temp_output_file)
        
        except KeyboardInterrupt:
            logger.warning("⚠️  KeyboardInterrupt detected during task launch!")
            raise
        
        if active_tasks:
            logger.info(f"⏳ Waiting for {len(active_tasks)} remaining tasks (max 2 min)...")
            try:
                remaining_results = await asyncio.wait_for(
                    asyncio.gather(*active_tasks, return_exceptions=True),
                    timeout=120.0
                )
                for result in remaining_results:
                    if isinstance(result, ConversationData):
                        results.append(result)
            except asyncio.TimeoutError:
                logger.warning(f"⚠️  Timeout! Cancelling {len(active_tasks)} stuck tasks")
                for task in active_tasks:
                    task.cancel()
        
        logger.info(f"✅ Successfully processed {len(results)}/{len(task_inputs)} conversations")
        return results
    
    async def _process_single_realtime(self, conversation: Dict, text: str, max_retries: int = 2) -> Optional[ConversationData]:
        """Process single conversation with minimal retries"""
        
        prompt = self.create_classification_prompt(text)
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=5,
                    logprobs=True,
                    top_logprobs=20,
                    timeout=20.0
                )
                
                if not response.choices:
                    return None
                
                content = response.choices[0].message.content.strip().upper()
                
                openai_class = None
                for token in self.valid_class_tokens:
                    if token in content:
                        openai_class = token
                        break
                
                if not openai_class:
                    letter_match = re.search(r'\b([A-M])\b', content)
                    if letter_match:
                        openai_class = letter_match.group(1)
                
                if not openai_class:
                    return None
                
                raw_tokens = []
                raw_logprobs = []
                
                if (response.choices[0].logprobs and 
                    response.choices[0].logprobs.content and
                    len(response.choices[0].logprobs.content) > 0):
                    
                    first_token_logprobs = response.choices[0].logprobs.content[0]
                    
                    if first_token_logprobs.top_logprobs:
                        for alt_token in first_token_logprobs.top_logprobs:
                            token_text = alt_token.token.strip().upper()
                            if token_text in self.valid_class_tokens:
                                raw_tokens.append(token_text)
                                raw_logprobs.append(alt_token.logprob)
                    else:
                        token_text = first_token_logprobs.token.strip().upper()
                        if token_text in self.valid_class_tokens:
                            raw_tokens.append(token_text)
                            raw_logprobs.append(first_token_logprobs.logprob)
                
                prob_map = self.filter_and_softmax_logprobs(raw_tokens, raw_logprobs)
                
                ordered_probs = [prob_map.get(token, 0.0) for token in self.all_valid_tokens_sorted]

                gpt4o_class = conversation.get('intent', 'unknown')
                agreement = (openai_class == gpt4o_class)

                return ConversationData(
                    chat_id=conversation.get('chat_id', f"unknown_{hash(text) % 10000}"),
                    sequence=conversation.get('sequence', text),
                    sequence_truncated=text,
                    gpt4o_label=gpt4o_class,
                    openai_label=openai_class,
                    openai_logprobs=ordered_probs,
                    openai_tokens=self.all_valid_tokens_sorted,
                    agreement=agreement
                )
            
            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
                
            except (openai.APITimeoutError, asyncio.TimeoutError):
                return None
                
            except Exception:
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(0.5)
        
        return None

def create_label_distribution_chart(results: List[ConversationData], save_path: str):
    label_counts = Counter([r.openai_label for r in results])
    labels = sorted(label_counts.keys())
    counts = [label_counts[label] for label in labels]
    
    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts)
    plt.title(f'Label Distribution (n={len(results)})')
    plt.xlabel('Predicted Label')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    
    for i, (label, count) in enumerate(zip(labels, counts)):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_distillation_data(results: List[ConversationData], output_path: str):
    distillation_data = []
    
    for conv in results:
        distillation_data.append({
            'chat_id': conv.chat_id,
            'text': conv.sequence_truncated,
            'hard_label': conv.gpt4o_label,
            'soft_labels': conv.openai_logprobs,
            'class_order': conv.openai_tokens,
            'teacher_prediction': conv.openai_label,
            'agreement': conv.agreement
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"distillation_ready": distillation_data}, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved {len(distillation_data)} training samples")

def load_conversations(filepath: str) -> List[Dict]:
    logger.info(f"📂 Loading: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        conversations = data
    else:
        conversations = [data]
    
    logger.info(f"✅ Loaded {len(conversations)} conversations")
    return conversations

async def main():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY not found")
        return
    
    INPUT_FILE = os.getenv("INPUT_FILE", "data/cleaned_sequences.json")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", "data/distillation_data_gpt4o_mini.json")
    DISTRIBUTION_CHART = os.getenv("DISTRIBUTION_CHART_PATH", "label_distribution.png")
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))
    TEMP_OUTPUT_FILE = "temp_results.json"

    logger.info("🚀 OpenAI GPT-4o-mini Labeler")
    
    try:
        conversations = load_conversations(INPUT_FILE)
    except FileNotFoundError:
        logger.error(f"❌ File not found: {INPUT_FILE}")
        return
    
    if not conversations:
        logger.error("❌ No conversations loaded")
        return

    results = []
    if os.path.exists(TEMP_OUTPUT_FILE):
        logger.info(f"📂 Loading partial results from {TEMP_OUTPUT_FILE}")
        with open(TEMP_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            try:
                partial_results_data = json.load(f)
                results = [ConversationData(**data) for data in partial_results_data]
                logger.info(f"✅ Loaded {len(results)} partial results")
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"⚠️  Could not decode partial results from {TEMP_OUTPUT_FILE}. Starting from scratch.")
                results = []

    processed_chat_ids = {res.chat_id for res in results}
    conversations_to_process = [conv for conv in conversations if conv.get('chat_id') not in processed_chat_ids]

    if not conversations_to_process:
        logger.info("✅ All conversations have already been processed.")
    else:
        labeler = OpenAILabeler(OPENAI_API_KEY, model=MODEL)
        
        start_time = time.time()
        
        try:
            new_results = await labeler.process_with_realtime_api(
                conversations_to_process, 
                max_concurrent=MAX_CONCURRENT,
                temp_output_file=TEMP_OUTPUT_FILE
            )
            results.extend(new_results)
        except KeyboardInterrupt:
            logger.warning("\n⚠️  KeyboardInterrupt! Saving partial results...")
        finally:
            elapsed_time = time.time() - start_time
            
            if results:
                logger.info(f"⏱️  Total time: {elapsed_time/60:.1f}m ({len(results)/elapsed_time:.1f} req/s)")
                create_label_distribution_chart(results, DISTRIBUTION_CHART)
                save_distillation_data(results, OUTPUT_FILE)
                logger.info("✅ Complete!")
            else:
                logger.error("❌ No results to save")

    if os.path.exists(TEMP_OUTPUT_FILE):
        os.remove(TEMP_OUTPUT_FILE)

if __name__ == "__main__":
    asyncio.run(main())
