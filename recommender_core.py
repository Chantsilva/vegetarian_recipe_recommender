import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load cleaned dataset
df_clean = pd.read_csv("recipes_cleaned_final.csv")

# 2. Build ingredients_text
if "ingredients_list" in df_clean.columns:
    def list_to_text(lst):
        if isinstance(lst, str):
            try:
                lst = ast.literal_eval(lst)
            except:
                lst = []
        return " ".join([str(x).lower() for x in lst])
    df_clean["ingredients_text"] = df_clean["ingredients_list"].apply(list_to_text)
else:
    df_clean["ingredients_text"] = df_clean["ingredients"].astype(str).str.lower()

# 3. TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words="english")
X_ingredients = vectorizer.fit_transform(df_clean["ingredients_text"])

def recommend_recipes(user_ingredients: str, top_n: int = 5):
    """
    Recebe ingredientes do utilizador (string com vírgulas)
    e devolve top_n receitas mais semelhantes pelos ingredientes.
    """
    query_text = " ".join(
        [x.strip().lower() for x in user_ingredients.split(",") if x.strip()]
    )
    if not query_text:
        return df_clean.head(0)

    query_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(query_vec, X_ingredients)[0]

    top_idx = sims.argsort()[::-1][:top_n]
    results = df_clean.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]
    return results[["recipe_title", "category", "ingredients_text", "similarity"]]