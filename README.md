
# Recipe Agent using Google ADK

A Recipe Suggestion & Research Agent built with **Google’s Agent Development Kit (ADK)**. This project demonstrates how to use ADK to build an intelligent multi‑agent workflow that takes ingredients as input and returns recipe recommendations and related information. :contentReference[oaicite:0]{index=0}

## 🚀 Features

- 🤖 **AI‑Powered Agents** – Uses ADK agents to process input and generate recipe outputs  
- 📋 **Recipe Generation** – Suggests recipes based on provided ingredients  
- 🔎 **Research & Insights** – Looks up background info like nutrition, cooking tips, and variations  
- 🛠️ **Modular Architecture** – Agents broken into components for easy improvement

## 📦 Prerequisites

Before you begin, ensure you have:

- Python **3.9+**
- A **Google API key** for Gemini models
- (Optional) Tools or APIs for nutrition research

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sisira214/Recipe-agent-using-ADK.git
   cd Recipe-agent-using-ADK


2. **Create and activate a Python virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate     # macOS/Linux
   .\.venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```



## 📁 Project Structure

```text
.
├── Data/                        # Your data files (ingredients etc.)
├── recipie/                    # Agent implementation code
├── requirements.txt            # Python dependencies
├── .env.example                # Sample environment config
└── README.md
```

> The `recipie/` directory contains the core agent workflows and logic for recipe generation.

## ▶️ Running the Agent

Use the ADK CLI to start the workflow locally:

```bash
adk web -port 8010
```




## ✨ Example Interaction

```
Input: "I have chicken, rice, and broccoli"
Output:
- Lemon garlic chicken stir fry
- Chicken rice bowl with steamed broccoli
- Suggested prep time: 30 min
- Nutrition facts…
```

*(Replace with actual example outputs from your agent.)*



## 📝 Contributing

Thanks for considering contributing! Please open issues or submit pull requests for:

* Bugs / errors
* Feature improvements
* Documentation updates

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.



[1]: https://gist.github.com/vincentkoc/638f11fcb00d0351473be5a8705d4c08?utm_source=chatgpt.com "Google Cloud/Gemini ADK + Opik for Agent ADK · GitHub"
