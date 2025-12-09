import streamlit as st
from recommender_core import recommend_recipes

st.set_page_config(
    page_title="Vegetarian Recipe Recommender",
    page_icon="🥕",
    layout="centered",
)

st.title("🥕 Vegetarian Recipe Recommender (CSV)")
st.write(
    "Type the ingredients you have at home (comma-separated) and I will "
    "suggest vegetarian recipes from the dataset."
)

ingredients_input = st.text_area(
    "Ingredients (comma-separated):",
    "tomatoes, onion, garlic, olive oil, pasta, cheese",
    height=80,
)

top_n = st.slider(
    "Number of recipes to show:",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
)

if st.button("Find recipes"):
    with st.spinner("Searching recipes..."):
        results = recommend_recipes(ingredients_input, top_n=top_n)

    if results.empty:
        st.warning("Please provide at least one ingredient.")
    else:
        for i, row in results.iterrows():
            st.subheader(f"{row['recipe_title']} (score: {row['similarity']:.3f})")
            st.write(f"**Category:** {row['category']}")
            st.write(f"**Ingredients (tokens):** {row['ingredients_text']}")
            st.markdown("---")





