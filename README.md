# RPI Landmarks Chatbot

We chose Option 2: GPT-enhanced chatbot
We give detailed information about 5 major RPI landmarks:
  - Russell Sage Laboratory
  - West Hall
  - RPI Union
  - Folsom Library
  - EMPAC


## File Descriptions

- `main.py`: Handles the user interface, including the welcome message, input loop, and error handling
- `chatbot.py`: Contains the RPIChatbot class with all the logic for:
  - Processing user input
  - Fuzzy matching for landmark identification
  - Handling follow-up questions
  - Generating responses about history, architecture, and current use
- `knowledge_base.json`: Structured data about each landmark including:
  - Historical information and timelines
  - Architectural features and styles
  - Current facilities and uses
  - Events and activities

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create .env file:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key if using GPT features
```

## Usage

Run the chatbot:
```bash
python main.py
```

Example conversation:
```
Welcome to the RPI History Chatbot!
Ask me anything about RPI's Landmarks.
Type 'quit' to exit.

You: Tell me about EMPAC
Bot: The Experimental Media and Performing Arts Center was opened in 2008. It is notable for being a state-of-the-art performing arts and media research center. Would you like to know more about its history, architecture, or current use?

You: Tell me about its architecture
Bot: EMPAC features Modern architecture. Notable architectural features include: Distinctive curved glass and steel exterior, Hillside integration design, Advanced acoustical engineering, Multi-story atrium.
```