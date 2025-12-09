# 🥦 Vegetarian Recipe Recommender

Turn **leftovers in the fridge** into **vegetarian meals** in a few seconds.

This project is an ingredient‑based recipe recommender built as the final project for a Data Science Bootcamp.  
Given a list of ingredients, it searches thousands of vegetarian recipes and returns the ones that best match what the user already has at home.

---

## ✨ Project Goals

- Help users decide *what to cook* using ingredients they already have.  
- Focus on **strictly vegetarian** recipes.  
- Build a **fast, interpretable** recommendation engine using classic NLP (no heavy deep learning).  
- Expose the model through a **simple Streamlit web app**.

---

## 📊 Dataset

The project uses a scraped / curated dataset of vegetarian recipes with:

- Recipe title and category  
- Full list of ingredients  
- Step‑by‑step instructions  
- Number of ingredients (`num_ingredients`)  
- Number of steps (`num_steps`)

Due to GitHub size limits, the full cleaned dataset  
`recipes_cleaned_final.csv` (≈73 MB, 31k recipes) is **not stored in this repo**.  
It is provided separately as part of the bootcamp materials and can be placed in the project root when running the notebooks and app locally.

---

## 🧹 Data Cleaning & Preprocessing

Main steps in the cleaning pipeline:

1. **Load raw dataset**  
   - Inspect shape, columns, missing values and basic statistics.

2. **Remove duplicates and invalid rows**  
   - Drop perfect duplicates.  
   - Filter out recipes with very few ingredients or steps.

3. **Parse and standardise text fields**  
   - Convert JSON‑like strings for `ingredients` and `directions` into Python lists.  
   - Lowercase, remove punctuation and normalise ingredient text.

4. **Create helper columns**  
   - `ingredients_list`, `directions_list`  
   - `ingredient_tokens` for easier matching  
   - `num_ingredients`, `num_steps` for analysis.

The final cleaned DataFrame is saved as `recipes_cleaned_final.csv` and used by both EDA and the recommender.

---

## 🧠 Recommendation Engine (TF‑IDF + Cosine Similarity)

The core of the system is a classic information‑retrieval model:

1. **Recipe representation**  
   - Use `TfidfVectorizer` on the cleaned ingredient text.  
   - Each recipe becomes a TF‑IDF vector that highlights its most characteristic ingredients.

2. **User query processing**  
   - The user types ingredients as a comma‑separated list.  
   - The input is normalised and transformed with the same vectorizer into a query vector.

3. **Similarity‑based ranking**  
   - Compute cosine similarity between the query vector and every recipe vector.  
   - Sort recipes by similarity score in descending order.  
   - Return the top‑N recipes as recommendations.

This approach keeps the model simple, explainable and fast enough to search the whole dataset in under a second on a standard laptop.

---

## 💻 Streamlit Web App

An interactive Streamlit app wraps the model:

- Text input for **available ingredients** (comma‑separated).  
- Slider for the **number of recipes** to return.  
- Button to trigger the search.  
- Results table showing:
  - recipe title  
  - category  
  - similarity score  
  - short ingredient snippet

### Run the app locally

From the project root:


Then open the URL shown in the terminal (typically `http://localhost:8501`).

---

## 📈 Exploratory Data Analysis (EDA)

The EDA notebook explores the cleaned dataset and answers questions like:

- **Most common ingredients** across vegetarian recipes.  
- **Top categories** (e.g. soups, pasta, salads, casseroles).  
- Relationship between **number of ingredients** and **number of steps**.  
- A simple **complexity score** combining ingredients and steps.

Key plots are exported as `eda_plots.png` and used in the project presentation.

---

## 🗂 Project Structure

├── app.py # Streamlit web interface
├── recommender_core.py # TF-IDF + cosine recommendation logic
├── notebooks
│ ├── 01_data_cleaning_and_eda.ipynb
│ └── 02_tfidf_recommender.ipynb
├── requirements.txt
└── README.md


(Names may differ slightly; adjust this section to match your actual layout.)

---

## 🚀 How to Use

1. Download / place `recipes_cleaned_final.csv` in the project root.  
2. Install the Python dependencies with `pip install -r requirements.txt`.  
3. Run `streamlit run app.py`.  
4. Type the ingredients you have at home and explore the top‑N vegetarian recipes ranked by similarity.

---

## 🔮 Future Work

- Incorporate **nutrition information and cooking time** into the ranking.  
- Add **personalisation** based on user preferences and past interactions.  
- Experiment with a lightweight **GenAI layer** to:
  - generate friendly recipe descriptions  
  - suggest ingredient substitutions when something is missing.
