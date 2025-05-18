import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from downloader import download_and_extract
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Example usage:
    zip_url = "https://huggingface.co/kushaaagr/Vilt-finetuned-for-VQA/resolve/main/vilt-finetuned-vqa-15.zip"
    output_directory = "."
    extracted_path = download_and_extract(zip_url, output_directory)

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)

    # Load model and processor, move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Load processor and config from finetuned LoRA folder
    processor = ViltProcessor.from_pretrained("vilt-finetuned-vqa-15")
    config = ViltConfig.from_pretrained("vilt-finetuned-vqa-15")  # Must include correct num_labels = 841

    # ✅ Load base model with config — but DO NOT load weights from vilt-finetuned-vqa
    base_model = ViltForQuestionAnswering.from_pretrained(
        "dandelin/vilt-b32-finetuned-vqa",
        config=config,
        ignore_mismatched_sizes=True
    )

    # ✅ Attach the LoRA adapter trained on top of this config
    model = PeftModel.from_pretrained(base_model, "vilt-finetuned-vqa-15")

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
        answer = str(answer).split()[0].lower()
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()