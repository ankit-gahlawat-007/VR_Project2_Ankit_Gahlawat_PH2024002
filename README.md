
# 📘 Comprehensive Report: Visual Question Answering on ABO Dataset

## 1. **Data Curation**

We curated a Visual Question Answering (VQA) dataset using the **Amazon Berkeley Objects (ABO)** dataset. The goal was to generate diverse, image-grounded, single-word question-answer pairs using both product images and associated metadata.

### 🔧 Tools Used

- **Gemini 1.5 Flash** (via Google AI Studio): Used for generating questions based on image and metadata.
- **Python + Google SDK (`google-genai`)**: Handled multimodal prompts.
- **Metadata**: Parsed from `listings_1.json` and `images.csv` from the ABO dataset.

### 🧠 Prompt Design

We used a structured prompt to instruct the model to:

- Ask 3 image-grounded questions.
- Ensure answers were **single words** and **present in metadata**.
- Format the response as a valid JSON array of Q-A pairs.

Example prompt snippet:
```json
[
  {
    "question": "What color is the blender?",
    "answer": "white"
  },
  {
    "question": "What material is the blender made of?",
    "answer": "plastic"
  }
]
```

### 🔁 Process Pipeline

1. Randomly sampled **1024** product listings from ABO.
2. Merged listings with image metadata to obtain image paths and dimensions.
3. Extracted metadata fields like `color`, `material`, `brand`, `product_type`, etc.
4. For each product image:
   - Read the image in binary.
   - Fused it with metadata into a multimodal Gemini prompt.
   - Parsed the returned JSON of 3 Q-A pairs.
   - Stored all output in timestamped JSON files.

### 🛠 Sample Output Format

Each curated sample includes:
```json
{
  "image_id": "abc123",
  "image_path": "/images/small/ab/abc123.jpg",
  "question": "What is the color of the object?",
  "answer": "red"
}
```

---

## 2. **Model Choices**

We selected `dandelin/vilt-b32-finetuned-vqa` for its following strengths:

- ⚡ **Efficient Vision-Language Processing**: Does not rely on region-based features.
- 🔍 **Pre-trained for VQA**: Already trained on VQA v2, offering good generalization.
- 🪶 **Compatible with LoRA**: Supports parameter-efficient tuning via `peft`.

### ❓Alternatives Considered

| Model              | Reason for Rejection                         |
|-------------------|----------------------------------------------|
| `BLIP-2`           | Too large for Kaggle's compute limitations.  |
| `CLIP`             | Not directly suited for Q-A without retraining. |
| `LLaVA`            | Overkill for single-word answers.            |

---

## 3. **Fine-Tuning Approach**

### 📦 Base Model

- `dandelin/vilt-b32-finetuned-vqa` (from HuggingFace)
- `ViltProcessor` used for tokenization and image preprocessing.

### ⚙️ Custom Dataset

We created a `CustomVQADataset` that:
- Resizes input image to 384×384.
- Uses the processor to tokenize text and image.
- Converts the label into one-hot encoding over the answer space.

### 🧩 LoRA Configuration

Using `peft`:
```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
```

LoRA enabled efficient fine-tuning of just a few trainable parameters within attention layers.

### 🔁 Training Loop

- Optimizer: `AdamW`, learning rate `5e-5`
- Epochs: `5`
- Batch Size: `16`
- Device: GPU (`cuda`) when available

The training loop computes loss via the model's classification head using one-hot labels, backpropagates, and optimizes parameters.

---

## 4. **Evaluation Metrics**

> _(Leave this section blank for now. Fill in later after running the evaluation.)_

### ✅ Standard Metrics

- **Accuracy**:
- **F1 Score**:

### 📌 Additional Metrics (Planned)

- **BERTScore** or **BLEU** for answer similarity.
- **Top-k Accuracy** in case of ambiguous answers.

---

## 5. **Any Additional Contribution / Novelty**

- ✅ **Prompt-grounded Question Generation** with Gemini using structured JSON output.
- ✅ **Metadata-anchored Q-A pairs** ensure interpretability.
- ✅ **Efficient Model Selection** compatible with Kaggle constraints (<7B params).
- ✅ **LoRA applied to Transformer-Vision model**, minimizing training footprint.
