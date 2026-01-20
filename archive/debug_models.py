import os
from dotenv import load_dotenv

# SDK Imports
from google import genai
from google.genai import types
from openai import OpenAI

# Load environment variables
load_dotenv()

def debug_gemini():
    print("\n--- 1. Testing Google Gemini API ---")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: 'GEMINI_API_KEY' not found in .env")
        return

    print(f"API Key found: {api_key[:5]}...{api_key[-4:]}")
    try:
        client = genai.Client(api_key=api_key)
        print("Sending request to 'gemini-2.5-flash-lite'...")
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents="Hello! Reply with 'Gemini OK'.",
        )
        print(f"SUCCESS: {response.text.strip()}")
    except Exception as e:
        print(f"FAILURE: {str(e)}")

def debug_gpt():
    print("\n--- 2. Testing OpenAI GPT API ---")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: 'OPENAI_API_KEY' not found in .env")
        return

    print(f"API Key found: {api_key[:5]}...{api_key[-4:]}")
    try:
        client = OpenAI(api_key=api_key)
        print("Sending request to 'gpt-5-nano'...")
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Hello! Reply with 'GPT OK'."}],
        )
        print(f"SUCCESS: {response.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"FAILURE: {str(e)}")

if __name__ == "__main__":
    print("Starting Model Connection Test...")
    debug_gemini()
    debug_gpt()
    print("\nTest Complete.")
