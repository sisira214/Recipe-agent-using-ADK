from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.llm_agent import Agent

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import re
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- CONFIG ----------------

RECIPE_CSV_PATH = "C:/Users/sashi/OneDrive/Documents/langgraphProjects/a2aRecipie/Data/recipies.csv"

FRIDGE_ITEMS = {
    "chicken", "rice", "vegetables", "pasta", "cheese", "eggs"
}

# ---------------- DATA ----------------

def load_recipe_db():
    df = pd.read_csv(RECIPE_CSV_PATH)

    recipes = []
    for _, row in df.iterrows():
        # Ensure no missing fields
        ingredients = [i.strip().lower() for i in str(row.get("ingredients", "")).split(",") if i]
        dietary = [d.strip().lower() for d in str(row.get("dietary", "none")).split(",") if d]
        cuisine = str(row.get("cuisine", "")).strip().lower()
        cooking_time = int(row.get("cooking_time", 999))

        # Provide default quantities and units if missing
        quantities = str(row.get("ingredient_quantities", "")).split(",")
        if len(quantities) != len(ingredients):
            quantities = ["1"] * len(ingredients)  # default 1

        units = str(row.get("ingredient_units", "")).split(",")
        if len(units) != len(ingredients):
            units = ["unit"] * len(ingredients)  # default "unit"

        recipes.append({
            "name": str(row.get("name", "Unknown")),
            "ingredients": ingredients,
            "dietary": dietary,
            "cuisine": cuisine,
            "cooking_time": cooking_time,
            "ingredient_quantities": quantities,
            "ingredient_units": units,
        })
    return recipes


def get_user_text(state) -> str:
    if not state:
        return ""

    if isinstance(state, dict) and state.get("input"):
        return str(state["input"]).lower()

    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if isinstance(last, dict):
            return str(last.get("content", "")).lower()

    return ""


# ---------------- TOOLS (ORDER UNCHANGED) ----------------

def extract_ingredients(state: dict | None = None, **kwargs):
    if state is None:
        state = {}
    text = get_user_text(state)
    state["ingredients"] = [i for i in FRIDGE_ITEMS if i in text]
    return state


def extract_preferences(state: dict | None = None, **kwargs):
    if state is None:
        state = {}
    text = get_user_text(state)

    dietary = []
    if "vegetarian" in text:
        dietary.append("vegetarian")
    if "vegan" in text:
        dietary.append("vegan")

    # If no dietary preference is given, default to 'none'
    if not dietary:
        dietary = ["none"]

    time_match = re.search(r"(\d+)\s*minutes?", text)

    state["dietary_restrictions"] = dietary
    state["max_cooking_time"] = int(time_match.group(1)) if time_match else 999
    state["cuisine_preference"] = (
        "indian" if "indian" in text else
        "italian" if "italian" in text else
        "asian" if "asian" in text else ""
    )
    return state


def search_recipes(state: dict | None = None, **kwargs):
    if state is None:
        state = {}
    recipes = load_recipe_db()

    state_ing = set(state.get("ingredients", []))
    state_diet = set(state.get("dietary_restrictions", ["none"]))
    cuisine = state.get("cuisine_preference", "").lower()
    max_time = state.get("max_cooking_time", 999)

    results = []

    for r in recipes:
        if r["cooking_time"] > max_time:
            continue
        if state_diet and not state_diet.issubset(set(r["dietary"])):
            continue
        if state_ing and set(r["ingredients"]).isdisjoint(state_ing):
            continue
        if cuisine and r["cuisine"] != cuisine:
            continue
        results.append(r)

    state["matched_recipes"] = results
    return state


def rank_recipes(state: dict | None = None, **kwargs):
    if state is None:
        state = {}
    recipes = state.get("matched_recipes", [])
    recipes.sort(
        key=lambda r: len(set(r["ingredients"]).intersection(state.get("ingredients", []))),
        reverse=True
    )
    state["matched_recipes"] = recipes
    return state


def check_fridge_availability(state: dict | None = None, **kwargs):
    if state is None:
        state = {}
    if not state.get("matched_recipes"):
        state["missing_ingredients"] = []
        return state

    recipe = state["matched_recipes"][0]
    missing = []

    for ing, qty, unit in zip(
        recipe["ingredients"],
        recipe["ingredient_quantities"],
        recipe["ingredient_units"]
    ):
        if ing not in FRIDGE_ITEMS:
            missing.append({
                "ingredient": ing,
                "quantity": qty,
                "unit": unit
            })

    state["missing_ingredients"] = missing
    return state


def generate_recommendation(state: dict | None = None, **kwargs):
    if state is None:
        state = {}
    if not state.get("matched_recipes"):
        state["response"] = "❌ No recipes found matching your preferences."
        return state

    r = state["matched_recipes"][0]

    response = (
        f"🍛 {r['name']}\n"
        f"🌍 Cuisine: {r['cuisine'].title()}\n"
        f"⏱ {r['cooking_time']} minutes\n\n"
        f"Ingredients:\n"
    )

    for i, q, u in zip(r["ingredients"], r["ingredient_quantities"], r["ingredient_units"]):
        response += f"- {i}: {q} {u}\n"

    state["selected_recipe"] = r
    state["response"] = response
    return state


def generate_cooking_steps_llm(state: dict | None = None, **kwargs) -> str:
    if state is None:
        state = {}

    # Pick first matched recipe if nothing is explicitly selected
    recipe = state.get("selected_recipe") or (state.get("matched_recipes") or [None])[0]

    if not recipe:
        return "❌ No recipe available to generate cooking steps."

    ingredient_text = "\n".join(
        [
            f"- {i}: {q} {u}"
            for i, q, u in zip(
                recipe["ingredients"],
                recipe.get("ingredient_quantities", ["1"] * len(recipe["ingredients"])),
                recipe.get("ingredient_units", ["unit"] * len(recipe["ingredients"])),
            )
        ]
    )

    prompt = f"""
You are a professional chef.

Generate step-by-step cooking instructions for the following recipe.

STRICT RULES:
- Use ONLY the ingredients listed below
- Do NOT add or assume any extra ingredients
- Do NOT mention ingredients not listed
- Follow Indian cooking style if cuisine is Indian
- Provide clear numbered steps

Recipe Name: {recipe['name']}
Cuisine: {recipe['cuisine']}
Cooking Time: {recipe['cooking_time']} minutes

Ingredients:
{ingredient_text}

Output only the cooking steps.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You generate cooking instructions from structured data only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3
    )

    steps = response.choices[0].message.content.strip()

    return f"""
🍽️ Recipe: {recipe['name']}
🌍 Cuisine: {recipe['cuisine']}
⏱️ Cooking Time: {recipe['cooking_time']} minutes

🧾 Ingredients:
{ingredient_text}

👩‍🍳 Cooking Instructions:
{steps}
"""

'''
def generate_cooking_steps_llm(state: dict | None = None, **kwargs):
    recipe = state.get("selected_recipe")
    if not recipe:
        return state

    ingredient_text = "\n".join(
        f"- {i}: {q} {u}"
        for i, q, u in zip(
            recipe["ingredients"],
            recipe["ingredient_quantities"],
            recipe["ingredient_units"]
        )
    )

    prompt = f"""
Generate step-by-step cooking instructions.

RULES:
- Use ONLY listed ingredients
- No new ingredients
- Numbered steps
- Indian cooking style

Recipe: {recipe['name']}
Ingredients:
{ingredient_text}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You generate cooking steps from structured data only."},
            {"role": "user", "content": prompt}
        ]
    )

    steps = completion.choices[0].message.content.strip()

    state["response"] += f"\n\nCooking Steps:\n{steps}"
    return state
'''
# ---------------- AGENTS ----------------

llm = LiteLlm(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

recipe_agent = Agent(
    name="recipe_recommender_agent",
    model=llm,
    description="Recommends recipes from CSV only",
    tools=[
        extract_ingredients,
        extract_preferences,
        search_recipes,
        rank_recipes,
        check_fridge_availability,
        generate_recommendation,
        generate_cooking_steps_llm,
    ],
    instruction="""
You are a Recipe Recommendation Agent. You help users find recipes based on ingredients and dietary preferences.
Follow these internal steps:
1. load_recipe_db() - Load the recipe database from CSV.
2. extract_ingredients(state) - Find ingredients mentioned in the user message.
3. extract_preferences(state) - Extract dietary restrictions, max cooking time, and cuisine preferences.
4. search_recipes(state) - Find matching recipes in the database.
5. Check_fridge_availability(state) - Identify missing ingredients from the fridge.
6. rank_recipes(state) - Order recipes by relevance.
7. generate_recommendation(state) - Create a detailed recipe recommendation response.
8. generate_cooking_steps_llm(recipe) - Generate cooking steps for the selected recipe.

CRITICAL RULES:
- You MUST call tools to compute the answer
- You MUST return the exact text from state["response"]
- You are NOT allowed to say "sorry" or invent explanations
- If matched_recipes is empty, say so ONLY via generate_recommendation


Output MUST include:
- matched_recipes
- missing_ingredients

Only answer recipe-related queries.
Do not invent recipes or ingredients.
Use only the CSV database.

Only answer recipe-related queries. Do not make up recipes or ingredients not in the database.The agent should use the tools in the specified order to process user requests effectively.It should take into account dietary restrictions, cooking time, and cuisine preferences when recommending recipes. agent should respond in a friendly and helpful manner, providing clear instructions or suggestions based on the user's input.  Agent should not make up any recipes or ingredients that are not present in the database. It should provide the recipie in detail.
"""
)

root_agent = Agent(
    name="kitchen_root_agent",
    model=llm,
    sub_agents=[recipe_agent],
    instruction="Delegate recipe queries only."
)