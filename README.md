# Jarvis Mobile

A voice and text assistant that runs in Termux on Android. It uses the Gemini API with automatic function calling to control the phone: calls, messages, apps, brightness and more.

## Setup

1. Install Termux and Termux:API from F-Droid.
2. In Termux: `pkg install python termux-api` and `pip install google-genai python-dotenv`.
3. Create a `.env` file next to `jarvis.py`:

```
GEMINI_API_KEY=your_key
OWNER_NAME=your_name
```

4. Run `python jarvis.py`.

Built with AI-assisted development.
