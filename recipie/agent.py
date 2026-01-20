c
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
INGREDIENTS_CSV_PATH = "C:/Users/sashi/OneDrive/Documents/langgraphProjects/a2aRecipie/Data/ingredients.csv"



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

# ---------------- AGENT 2 DEFINITION ----------------
# After getting the ingredients that are not present in fridge we can call another agent to suggest nearby stores to buy those ingredients from. I have ingredients data in 
# form of csv file which contains the following columns: ingredient,store,price_per_unit,unit example data is rice,Walmart,1.2,grams

def load_ingredient_prices():
    return pd.read_csv(INGREDIENTS_CSV_PATH).to_dict(orient="records")




# ---------------- AGENT 2 TOOL ----------------

def add_missing_ingredients_to_cart(state: dict | None = None, **kwargs):
    """
    Agent 2: Market / Ingredient Purchase Agent

    Input (from Agent 1):
        state["missing_ingredients"] = [
            {"ingredient": "rice", "quantity": "200", "unit": "grams"}
        ]

    Output:
        state["cart"]
        state["total_cart_cost"]
    """

    # If nothing is missing, no need to shop
    if state is None:
        state = {}

    if state.get("missing_ingredients") is None:
        state["missing_ingredients"] = []

    if not state.get("missing_ingredients"):
        state["cart"] = []
        state["total_cart_cost"] = 0.0
        return state


    prices = load_ingredient_prices()

    cart = []
    total_cost = 0.0

    for item in state["missing_ingredients"]:
        ingredient = item["ingredient"]
        quantity = float(item["quantity"])
        unit = item["unit"]

        matches = [
            p for p in prices
            if p["ingredient"] == ingredient and p["unit"] == unit
        ]

        if not matches:
            continue

        best_option = min(matches, key=lambda x: float(x["price_per_unit"]))
        price_per_unit = float(best_option["price_per_unit"])
        cost = quantity * price_per_unit

        cart.append({
            "ingredient": ingredient,
            "store": best_option["store"],
            "quantity": quantity,
            "unit": unit,
            "price_per_unit": price_per_unit,
            "cost": round(cost, 2)
        })

        total_cost += cost

    state["cart"] = cart
    state["total_cart_cost"] = round(total_cost, 2)
    return state

# ---------------- OPTIONAL SUMMARY TOOL ----------------

def summarize_cart(state: dict | None = None, **kwargs):
    """
    Generates a human-readable cart summary.
    """

    if state is None:
        state = {}

    cart = state.get("cart", [])

    if not cart:
        state["cart_summary"] = "🛒 No ingredients need to be purchased."
        state["total_cart_cost"] = 0.0
        return state

    total_cost = state.get("total_cart_cost", 0.0)

    summary = "🛒 Ingredients added to cart:\n"

    for item in cart:
        summary += (
            f"- {item['ingredient']} from {item['store']}: "
            f"{item['quantity']} {item['unit']} × "
            f"{item['price_per_unit']} = ${item['cost']}\n"
        )

    summary += f"\n💰 Total Cost: ${total_cost}"

    state["cart_summary"] = summary
    state["total_cart_cost"] = total_cost
    return state

# ---------------- AGENT 3 DEFINITION ----------------
# Wallet Management Agent. This agent can help user manage their budget for grocery shopping based on their total cart cost from Agent 2. 

# ---------------- AGENT 3: WALLET MANAGEMENT AGENT ----------------

# Fixed wallet credentials
VALID_USER_ID = "0101"
VALID_PASSWORD = "wallet"

# Initial wallet balance
WALLET_BALANCE = 20000.0





def wallet_agent(state: dict | None = None, **kwargs) -> dict:
    """
    Wallet Management Agent

    Input (from Agent 2 / User):
        state["total_cart_cost"]       : float
        state["user_id"]               : str
        state["password"]              : str
        state["confirm_payment"]       : bool

    Output:
        state["transaction_status"]    : str
        state["remaining_balance"]     : float (if successful)
    """

    global WALLET_BALANCE

    if state is None:
        state = {}

    # Inject default credentials automatically if missing
    if "user_id" not in state:
        state["user_id"] = VALID_USER_ID
    if "password" not in state:
        state["password"] = VALID_PASSWORD
    if "confirm_payment" not in state:
        state["confirm_payment"] = True

    total_cost = state.get("total_cart_cost", 0.0)
    user_id = state["user_id"]
    password = state["password"]
    confirm = state["confirm_payment"]

    # No cart cost
    if total_cost <= 0.0:
        state["transaction_status"] = "NO_CART"
        return state

    # Authentication check
    if user_id != VALID_USER_ID or password != VALID_PASSWORD:
        state["transaction_status"] = "FAILED_AUTH"
        return state

    # Payment confirmation
    if not confirm:
        state["transaction_status"] = "CANCELLED"
        return state

    # Check wallet balance
    if WALLET_BALANCE < total_cost:
        state["transaction_status"] = "INSUFFICIENT_FUNDS"
        return state

    # Deduct the amount
    WALLET_BALANCE -= total_cost

    state["transaction_status"] = "SUCCESS"
    state["remaining_balance"] = WALLET_BALANCE

    return state

# ---------------- AGENTS DEFINITION ----------------

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


missing_items_toCart_agent = None
try:
    missing_items_toCart_agent = Agent(
        # Using a potentially different/cheaper model for a simple task
        model=LiteLlm(model="openai/gpt-4o-mini",api_key=os.getenv("OPENAI_API_KEY")),# Use LiteLlm to specify the model name,
    name='missing_items_toCart_agent',
    description="Tells the missing ingredients to shopping cart with prices",
    instruction="""
You are a Shopping Cart Agent. You help users add missing ingredients to their shopping cart and summarize the total cost.
Follow these internal steps:
1. load_ingredient_prices() - Load the ingredient prices from CSV.
2. add_missing_ingredients_to_cart(state) - Add missing ingredients to shopping carrt with prices based on ingredient prices.
3. summarize_cart(state) - Summarize the shopping cart with total cost.

Input:
- missing_ingredients from Recipe Agent

Output:
- cart
- total_cart_cost


Do not make up recipes or ingredients not in the database.The agent should use the tools in the specified order to process user requests effectively.
""",
    tools=[
    load_ingredient_prices, add_missing_ingredients_to_cart, summarize_cart
],
    )
    print(f"✅ Agent '{missing_items_toCart_agent.name}' created using model '{missing_items_toCart_agent.model}'.")
except Exception as e:
    print(f"❌ Could not create Greeting agent. Check API Key ({missing_items_toCart_agent.model}). Error: {e}")



wallet_management_agent = None
try:
    wallet_management_agent = Agent(
        # Using a potentially different/cheaper model for a simple task
        model=LiteLlm(model="openai/gpt-4o-mini",api_key=os.getenv("OPENAI_API_KEY")), # Use LiteLlm to specify the model name,
    name='rwallet_management_agent',
    description="Allows user to manage their wallet for grocery shopping",
    instruction="""
You are a Wallet Management Agent. You help users to order missing items by getting the total cost from agent 2.
Follow these internal steps:
1. wallet_agent(state) - Manage wallet balance and process payment for the shopping cart.

Input:
- total_cart_cost from Cart Agent

Authenticate user and process payment.

The agent should use the tools in the specified order to process user requests effectively. 
""",
    tools=[
    wallet_agent
],
    )
    print(f"✅ Agent '{missing_items_toCart_agent.name}' created using model '{missing_items_toCart_agent.model}'.")
except Exception as e:
    print(f"❌ Could not create Greeting agent. Check API Key ({missing_items_toCart_agent.model}). Error: {e}")
    

root_agent = None

if recipe_agent and missing_items_toCart_agent and wallet_management_agent:

    root_agent = Agent(
        name="kitchen_root_agent",
        model=LiteLlm(
            model="openai/gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        description="Coordinates recipe recommendation, shopping cart, and wallet payment agents",
        instruction="""
You are the Kitchen Root Orchestrator Agent.

Delegate tasks only.
When delegating, always pass user input as state['input'] to the sub-agent.
After sub-agents finish, summarize the result for the user.

Behavior:
- If user asks for a recipe → delegate to recipe_recommender_agent
- If missing ingredients are found → delegate to missing_items_toCart_agent
- If total cart cost is computed → delegate to wallet_management_agent

You MUST delegate tasks to sub-agents.
Do not perform tool logic yourself.
""",
        sub_agents=[
            recipe_agent,
            missing_items_toCart_agent,
            wallet_management_agent
        ]
    )

    print(
        f"✅ Root Agent '{root_agent.name}' created with sub-agents: "
        f"{[a.name for a in root_agent.sub_agents]}"
    )

else:
    print("❌ Root agent creation failed — one or more sub-agents missing.")


#I have chicken. Give me a Indian recipie
