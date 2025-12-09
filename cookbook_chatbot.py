# ================================
# 0. Install (run once in a terminal or notebook):
# pip install langchain langchain_community pypdf
# pip install termcolor langchain_openai langchain-huggingface sentence-transformers chromadb langchain_chroma tiktoken openai python-dotenv
# ================================

import os
import re
import warnings
from typing import List, Dict, Set, Tuple

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import pypdf
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


import openai
# ================================
# 1. Environment and API setup
# ================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment. Set it in a .env file.")
openai.api_key = OPENAI_API_KEY

client = openai.OpenAI()
# ================================
# 2. Load PDF and extract text
# ================================
PDF_PATH = "data/low_budget_vegetarian_cooking.pdf"

def load_pdf_pages(pdf_path: str) -> List[Document]:
    reader = pypdf.PdfReader(pdf_path)
    pages: List[Document] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(Document(page_content=text, metadata={"page": i + 1}))
    return pages

pages = load_pdf_pages(PDF_PATH)
print(f"Loaded {len(pages)} pages from PDF.")

# ================================
# 3. Very simple recipe extraction
#    (You can improve this later)
# ================================

def is_recipe_title(line: str) -> bool:
    """
    Heuristic: treat a line as a recipe title if:
    - It is not empty
    - Not all caps section heading like 'Bean Dishes'
    - Medium length
    """
    line = line.strip()
    if len(line) < 3:
        return False
    # Ignore TOC page numbers and obvious non-titles
    if line.lower().startswith("table of contents"):
        return False
    if re.match(r"^\d+\.?$", line):
        return False
    # Many recipe titles have capitalized words but not ALL CAPS
    # You can tune this rule.
    words = line.split()
    if len(words) > 8:
        return False
    # Avoid section headings by checking if all words are capitalized
    if all(w.isalpha() and w[0].isupper() for w in words):
        return True
    return False
def extract_recipes_from_pages(pages: List[Document]) -> List[Dict]:
    recipes = []
    current_recipe = None
    collecting_ingredients = False

    for page in pages:
        page_num = page.metadata.get("page", None)
        lines = page.page_content.split("\n")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Detect a new recipe title
            if is_recipe_title(stripped):
                # Save previous recipe if exists
                if current_recipe is not None:
                    recipes.append(current_recipe)
                current_recipe = {
                    "title": stripped,
                    "page": page_num,
                    "ingredients_raw": [],
                    "instructions_raw": ""
                }
                collecting_ingredients = True
                continue
            # Heuristic: stop ingredients and start instructions
            # if we see a long paragraph or a sentence with a period.
            if collecting_ingredients:
                # Lines that look like ingredient lines: contain quantities or short phrases
                if re.search(r"\d", stripped) or len(stripped.split()) <= 6:
                    current_recipe["ingredients_raw"].append(stripped)
                else:
                    # Switch to instructions
                    collecting_ingredients = False
                    current_recipe["instructions_raw"] += stripped + " "
            else:
                if current_recipe is not None:
                    current_recipe["instructions_raw"] += stripped + " "

    # Save last one
    if current_recipe is not None:
        recipes.append(current_recipe)

    # Filter out obviously bad extractions (no ingredients)
    recipes = [r for r in recipes if len(r["ingredients_raw"]) > 0]
    return recipes
recipes = extract_recipes_from_pages(pages)
print(f"Extracted {len(recipes)} candidate recipes (heuristic).")
# ================================
# 4. Ingredient normalisation
# ================================

def normalize_ingredient(text: str) -> str:
    text = text.lower()
    # Remove quantities (numbers and fractions)
    text = re.sub(r"\d+\/?\d*\s*", " ", text)
    # Remove units and common words
    text = re.sub(
        r"\b(cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|grams|g|kg|ml|l|ounce|ounces|oz)\b",
        " ",
        text
    )
    # Remove punctuation
    text = re.sub(r"[(),.:;]", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

CANONICAL_MAP = {
    "brown rice": "rice",
    "white rice": "rice",
    "basmati rice": "rice",
    "lentils": "lentils",
    "red lentils": "lentils",
    "yellow split peas": "split peas",
    "green split peas": "split peas",
    "olive oil": "oil",
    "vegetable oil": "oil",
    "onions": "onion",
    "carrots": "carrot",
    "tomatoes": "tomato",
    "diced tomatoes": "tomato",
    "canned tomatoes": "tomato",
}

def canonicalize(name: str) -> str:
    return CANONICAL_MAP.get(name, name)

def build_normalized_ingredients(recipes: List[Dict]) -> None:
    for r in recipes:
        norm_set: Set[str] = set()
        for raw in r["ingredients_raw"]:
            norm = normalize_ingredient(raw)
            if not norm:
                continue
            norm = canonicalize(norm)
            norm_set.add(norm)
        r["ingredients_normalized"] = norm_set

build_normalized_ingredients(recipes)

print("Example normalized ingredients for first 3 recipes:")
for r in recipes[:3]:
    print(r["title"])
    print(r["ingredients_raw"])
    print(r["ingredients_normalized"])
    print("-" * 40)
# ================================
# 5. Matching logic
# ================================

IGNORE_INGREDIENTS = {"water", "salt", "pepper", "oil"}  # can expand if needed

def parse_user_ingredients(user_str: str) -> Set[str]:
    parts = [p.strip() for p in user_str.split(",") if p.strip()]
    norm = set()
    for p in parts:
        n = normalize_ingredient(p)
        n = canonicalize(n)
        if n:
            norm.add(n)
    return norm

def match_score(recipe_set: Set[str], user_set: Set[str], ignore: Set[str]) -> float:
    core = recipe_set - ignore
    if not core:
        return 0.0
    missing = core - user_set
    return 1.0 - len(missing) / len(core)

def find_matching_recipes(user_ingredients_str: str, min_score: float = 0.0) -> List[Tuple[float, Dict]]:
    user_set = parse_user_ingredients(user_ingredients_str)
    results: List[Tuple[float, Dict]] = []
    for r in recipes:
        score = match_score(r["ingredients_normalized"], user_set, IGNORE_INGREDIENTS)
        results.append((score, r))
    results.sort(key=lambda x: x[0], reverse=True)
    return results

# ================================
# 6. Build vector DB for RAG
# ================================

def build_documents_for_rag(recipes: List[Dict]) -> List[Document]:
    docs = []
    for r in recipes:
        content = (
            f"Title: {r['title']}\n"
            f"Page: {r.get('page', 'unknown')}\n"
            f"Ingredients: {', '.join(r['ingredients_raw'])}\n\n"
            f"Instructions: {r['instructions_raw']}\n"
        )
        metadata = {
            "title": r["title"],
            "page": r.get("page", None),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

docs_for_rag = build_documents_for_rag(recipes)
db = Chroma.from_documents(
    docs_for_rag,
    embeddings,
    persist_directory="./chroma_low_budget_veg_recipes"
)
print("ChromaDB created with recipe documents.")
# ================================
# 7. Helper: format context from matched recipes
# ================================

def build_context_from_matches(matches: List[Tuple[float, Dict]]) -> str:
    ctx = ""
    for score, r in matches:
        ctx += f"\n=== RECIPE (score {score:.2f}) ===\n"
        ctx += f"Title: {r['title']}\n"
        ctx += f"Page: {r.get('page', 'unknown')}\n"
        ctx += "Ingredients:\n"
        for ing in r["ingredients_raw"]:
            ctx += f" - {ing}\n"
        ctx += "\nInstructions:\n"
        ctx += r["instructions_raw"] + "\n"
    return ctx.strip()
# ================================
# 8. Chat function
# ================================

MODEL_PARAMS = {
    "model": "gpt-4o",
    "temperature": 0.7,     # more creative
    "max_tokens": 800,      # you can increase if needed
    "top_p": 0.9,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.6,
}

def chat_with_cookbook(user_ingredients_str: str) -> str:
    """
    Main entry point:
    - user_ingredients_str: string like "rice, lentils, onion, carrot, tomato"
    - returns: chatbot answer as string
    """
    user_set = parse_user_ingredients(user_ingredients_str)
    if len(user_set) < 5:
        return (
            "Please provide at least 5 ingredients (comma-separated). "
            f"You provided: {', '.join(sorted(user_set)) if user_set else 'none'}."
        )
    matches = find_matching_recipes(user_ingredients_str, min_score=0.0)

    # Take always the top 3 matches
    top_matches = matches[:3]
    if not top_matches:
        return "I could not find any recipes in the cookbook."

    formatted_context = build_context_from_matches(top_matches)


    prompt = f"""
You are a helpful assistant that suggests low-budget vegetarian recipes
based on the cookbook 'Low-Budget Vegetarian Cooking'.

USER INGREDIENTS:
{', '.join(sorted(user_set))}

CONTEXT (recipes from the cookbook):
'''
{formatted_context}
'''

INSTRUCTIONS:
- Use ONLY the recipes in the context to answer.
- Suggest 1–3 recipes that best fit the user's ingredients.
- For each recipe, explain briefly:
  - Why it matches the given ingredients.
  - The main cooking idea in simple steps (in your own words, do not copy the book).
- If some recipes require 1–2 extra small ingredients (e.g., oil, salt, water, basic spices), you may mention them as small additions.
- Answer in clear English, friendly and practical.
"""

    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        messages=messages,
        **MODEL_PARAMS,
        timeout=120
    )
    answer = completion.choices[0].message.content
    return answer
# ================================
# 9. Simple CLI test
# ================================
if __name__ == "__main__":
    print("Low-Budget Vegetarian Cooking Chatbot")
    print("Type a comma-separated list of at least 5 ingredients (or 'quit' to exit).")
    while True:
        user_in = input("\nYour ingredients: ")
        if user_in.strip().lower() in {"quit", "exit"}:
            break
        reply = chat_with_cookbook(user_in)
        print("\n--- Chatbot answer ---")
        print(reply)
        print("----------------------")

