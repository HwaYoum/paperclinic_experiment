import os
from litellm import completion
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def debug_huggingface():
    print("--- Debugging Hugging Face Inference API ---")
    
    # 1. Check API Key
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        print("ERROR: 'HUGGINGFACE_API_KEY' is NOT set in your .env file.")
        print("Get your token here: https://huggingface.co/settings/tokens")
        return
    else:
        masked_token = f"{hf_token[:5]}...{hf_token[-4:]}"
        print(f"SUCCESS: Found HF Token: {masked_token}")

    # 2. Test Model (Llama-3-8B is generally available on HF Free Inference API)
    model_name = "huggingface/meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"\nAttempting to call model: {model_name}")
    print("Note: If this is your first time using this model, it might take a moment to load on HF servers.")

    try:
        response = completion(
            model=model_name,
            messages=[{"role": "user", "content": "Tell me a very short joke in Korean."}],
            api_key=hf_token
        )
        
        content = response.choices[0].message.content
        print(f"\n--- API SUCCESS! ---")
        print(f"Response from HF:\n> {content}")

    except Exception as e:
        print(f"\n--- API CALL FAILED ---")
        print(f"Error Message: {str(e)}")
        
        # Specific troubleshooting
        if "401" in str(e):
            print("Hint: 401 Unauthorized. Check if your token is valid and has 'Read' access.")
        elif "429" in str(e):
            print("Hint: 429 Too Many Requests. HF Free tier is rate-limited.")
        elif "503" in str(e):
            print("Hint: 503 Model is loading. Please wait a few seconds and try again.")
        elif "403" in str(e):
            print("Hint: 403 Forbidden. You may need to 'Accept' the model license on its HF model page (e.g., Llama-3 needs gated access approval).")

if __name__ == "__main__":
    debug_huggingface()
