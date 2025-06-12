from huggingface_hub import login, upload_folder
from dotenv import load_dotenv
load_dotenv()

def upload_to_hf(repo_id, save_dir):
    model.save_pretrained(save_dir)
    left_tokenizer.save_pretrained(save_dir)

    login(token=os.environ.get("HG_KEY_WRITE"))
    upload_folder(
        repo_id=repo_id,
        folder_path=save_dir,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', type=str, required=True, help='Hugging Face repository ID', 
                        default="lengoctuong/My-Huatuo")
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to upload')
    args = parser.parse_args()
    upload_to_hf(args.repo_id, args.save_dir)