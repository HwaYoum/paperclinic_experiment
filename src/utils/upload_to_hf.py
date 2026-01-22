import argparse
import os
from huggingface_hub import HfApi, login

def upload_files(repo_id, token=None, private=False):
    if token:
        print(f"Logging in with provided token...")
        login(token=token)
    
    api = HfApi()
    
    # 1. Create Repository if it doesn't exist
    print(f"Ensuring repository '{repo_id}' exists...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        print(f"Repository ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error creating/checking repository: {e}")
        return

    # Files to upload and their target names in the repo
    files_to_upload = {
        "data/refined_sentences.csv": "refined_sentences.csv",
        "data/converted_train.jsonl": "train.jsonl",
        "data/converted_val.jsonl": "validation.jsonl"
    }
    
    print(f"Starting upload to {repo_id}...")
    
    for local_path, repo_path in files_to_upload.items():
        if not os.path.exists(local_path):
            print(f"Warning: File not found: {local_path}. Skipping.")
            continue
            
        print(f"Uploading {local_path} as {repo_path}...")
        
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"Successfully uploaded {repo_path}")
        except Exception as e:
            print(f"Error uploading {repo_path}: {e}")

    print("Upload process complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload dataset files to Hugging Face Hub")
    parser.add_argument("repo_id", type=str, help="The Hugging Face repository ID (e.g., username/dataset)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (optional if already logged in)")
    parser.add_argument("--private", action="store_true", help="Make the repository private (only used if creating a new repo)")
    
    args = parser.parse_args()
    
    upload_files(args.repo_id, args.token, args.private)
