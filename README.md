# cs498-ai-team9

## Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate   # Windows

# Create .env and add your API key
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Install dependencies
pip install -r requirements.txt
```

Replace `your_api_key_here` in `.env` with your API key.
