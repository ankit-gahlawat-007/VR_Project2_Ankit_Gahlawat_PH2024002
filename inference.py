import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from downloader import download_and_extract
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
from peft import PeftModel

def normalize_answer(ans):
    ans = ans.lower()
    tokens = re.split(r'[_-]+', ans)
    final_tokens = []
    for token in tokens:
        camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', token).split()
        final_tokens.extend(camel_split)
    return ' '.join(final_tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Example usage:
    zip_url = "https://huggingface.co/kushaaagr/Vilt-finetuned-for-VQA/resolve/main/vilt-finetuned-vqa-15.zip"
    # output_directory = "."
    output_directory = f"{script_dir}/models"
    os.makedirs(output_directory, exist_ok=True)
    extracted_path = download_and_extract(zip_url, output_directory)
    model_dirname = "vilt-finetuned-vqa-15"
    vilt_dir = f"{output_directory}/{model_dirname}"

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)

    # Load model and processor, move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Load processor and config from finetuned LoRA folder
    processor = ViltProcessor.from_pretrained(vilt_dir)
    config = ViltConfig.from_pretrained(vilt_dir)  # Must include correct num_labels = 841

    # ✅ Load base model with config — but DO NOT load weights from vilt-finetuned-vqa
    base_model = ViltForQuestionAnswering.from_pretrained(
        "dandelin/vilt-b32-finetuned-vqa",
        config=config,
        ignore_mismatched_sizes=True
    )

    # ✅ Attach the LoRA adapter trained on top of this config
    model = PeftModel.from_pretrained(base_model, vilt_dir)

    model.to(device)
    model.eval()

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{args.image_dir}/{row['image_name']}"
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            encoding = processor(image, question, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
                predicted_idx = logits.argmax(-1).item()
                answer = model.config.id2label[predicted_idx]
        except Exception as e:
            answer = "error"
        # Ensure answer is one word and in English (basic post-processing)
        answer = normalize_answer(str(answer)).split()[0].lower()
        
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()