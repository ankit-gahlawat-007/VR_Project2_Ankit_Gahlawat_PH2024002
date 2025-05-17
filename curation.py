import time
import os
import sys
import json
import random
import csv
import re
import socket
import gzip

from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

from google.genai import types, Client
# If instead you installed "google-generativeai", comment out the above
# and uncomment these two lines:
# import google.generativeai as genai
# Client = genai

dotenv_path = os.path.join(os.path.dirname(__file__), '.env') # Adjust path as needed
load_dotenv(dotenv_path)


# 1) Configure API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise EnvironmentError("Please set the GOOGLE_API_KEY environment variable")
client = Client(api_key=API_KEY)


def is_connected(host="8.8.8.8", port=53, timeout=3):
    """
    Attempts to connect to the internet using a known public DNS (Google).
    Returns True if online, False if not.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False


# 2) Filepaths and parameters
# LISTINGS_PATH    = r"C:\datasets\Amazon Berkeley Objects Dataset\abo-listings\listings\metadata\listings_1.json\listings_1.json"
# IMAGES_CSV_PATH  = r"C:\datasets\Amazon Berkeley Objects Dataset\images\metadata\images.csv"
# IMAGES_ROOT      = r"C:\datasets\Amazon Berkeley Objects Dataset\images\small"
# N_SAMPLES        = 2
# K_REQUESTS       = 15


# 2) Filepaths and parameters
BASE_DATASET_PATH = r"C:\datasets\Amazon Berkeley Objects Dataset"
LISTINGS_FILE     = "listings_d.json.gz"  # You can change this filename easily

LISTINGS_PATH     = os.path.join(BASE_DATASET_PATH, "abo-listings", "listings", "metadata", LISTINGS_FILE)
IMAGES_CSV_PATH   = os.path.join(BASE_DATASET_PATH, "images", "metadata", "images.csv")
IMAGES_ROOT       = os.path.join(BASE_DATASET_PATH, "images", "small")

# N_SAMPLES         = 2     # Number of samples to process
N_SAMPLES         = 1024 
K_REQUESTS        = 15    # Pause every K requests


seed_value = 42  # You can use any integer as the seed
random.seed(seed_value)

# 2.1) Validate paths
errors = []

if not os.path.isfile(LISTINGS_PATH):
    errors.append(f"❌ Listings file not found: {LISTINGS_PATH}")

if not os.path.isfile(IMAGES_CSV_PATH):
    errors.append(f"❌ Images CSV file not found: {IMAGES_CSV_PATH}")

if not os.path.isdir(IMAGES_ROOT):
    errors.append(f"❌ Image folder not found: {IMAGES_ROOT}")

if errors:
    print("\n".join(errors))
    print("Please set the correct file paths before proceeding.")
    sys.exit(1)
    # raise FileNotFoundError("Please set the correct file paths before proceeding.")



# 3) Load & sample the LDJSON listings from GZipped JSON with progress bar
listings = []
open_func = gzip.open if LISTINGS_FILE.endswith('.gz') else open
with open_func(LISTINGS_PATH, 'rt', encoding='utf-8') as f:
    for line in tqdm(f, desc="🔄 Unzipping & Reading Listings"):
        try:
            listings.append(json.loads(line))
        except json.JSONDecodeError:
            continue

subset = random.sample(listings, N_SAMPLES)

# print("Selected listings", subset)

# 4) Load image metadata CSV
images_meta = {}
with open(IMAGES_CSV_PATH, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        images_meta[row['image_id']] = row


# 5) Join CSV → listings
oddities = []
for item in subset:
    img_id = item.get("main_image_id")
    if "item_id" not in item:
        print("item_id not found")
    if img_id in images_meta:
        m = images_meta[img_id]
        item['img_path'] = os.path.join(IMAGES_ROOT, m['path'])
        # print("item[img_path]", item['img_path'])
        item['img_w']    = int(m['width'])
        item['img_h']    = int(m['height'])
    else:
        oddities.append(item['item_id'])
        print("item does not has image")

subset = list(filter(lambda item: item['item_id'] not in oddities, subset))

# print(dir(subset[0]))
print(subset[0].keys())
print(subset[0]['item_id'])

# imgless = 0
# for item in subset:
#     if "img_path" in item.keys():
#         print(item['img_path'])
#     else:
#         imgless += 1
#         print("no img_path for item", item)
# print("# items with no images =", imgless)

with open('selected_imgs.txt', 'w') as file:
    selected_imgs = [item['img_path'] for item in subset]
    file.write(str(selected_imgs))
    # file.write(response.text)



# 6) Metadata extractor
def extract_metadata(item):
    return {
        "item_name":    item.get('item_name',[{}])[0].get('value',''),
        "bullet_point": [bp.get('value','') for bp in item.get('bullet_point',[])],
        "color":        item.get('color',[{}])[0].get('value',''),
        "material":     item.get('material',[{}])[0].get('value',''),
        "product_type": item.get('product_type',[{}])[0].get('value',''),
        "style":        item.get('style',[{}])[0].get('value',''),
        "item_keywords":[kw.get('value','') for kw in item.get('item_keywords',[])],
        "brand":        item.get('brand',[{}])[0].get('value',''),
        "node_name":    item.get('node', [{}])[-1].get('node_name',''),
        "model_year":   item.get('model_year',[{}])[0].get('value',''),
        "height":       item.get('item_dimensions',{}).get('height',{}).get('value',''),
        "width":        item.get('item_dimensions',{}).get('width',{}).get('value',''),
        "length":       item.get('item_dimensions',{}).get('length',{}).get('value',''),
    }

# 7) Prompt template
PROMPT = """
You are an expert in visual question answering.

Given this product image and its metadata, create 3 questions that:
1. Can be answered by looking at the image.
2. Have a one-word answer.
3. The answer must be grounded in the image and reflect visual understanding.
4. The answer should match a field from the metadata.
5. Output 3 questions
6. Ensure diversity in question types and difficulty levels
7. Answer should be in ascii encoding only and make sure the formatting is consistent among answers.

The output should be in JSON only, and the JSON object should have the following structure:
[{{
  "question": "...",
  "answer": "..."
}},...]

Example output:
[
  {{
    "question": "What color is the blender?",
    "answer": "white"
  }},
  {{
    "question": "What material is the blender made of?",
    "answer": "plastic"
  }},
  {{
    "question": "What is this object?",
    "answer": "blender"
  }},
  ...
]


Metadata:
{metadata}
"""

def clean_json_string(input_string):
    cleaned_string = input_string.strip(' `').replace('json', '').replace('```', '')
    cleaned_string = cleaned_string.strip()
    return cleaned_string


# 8) Call Gemini & collect VQA pairs
vqa_data = []
ts1 = datetime.now().strftime("%Y%m%d_%H%M%S")
for i, item in enumerate(subset):
    # if True:
    #     continue
    img_path = item.get('img_path')
    if not img_path or not os.path.exists(img_path):
        continue

    # Sleep after every K requests
    if i > 0 and i % K_REQUESTS == 0:
        try:
            out_file = f"part {LISTINGS_FILE.removesuffix('.json.gz')} curated vqa data {ts1}.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(vqa_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(vqa_data)} entries to {out_file}")
        except (IOError, OSError, TypeError, ValueError) as e:
            print(f"❌ Failed to write VQA data to file: {e}")

        print(f"Sleeping for 61 seconds after {i} requests to avoid rate limits...")
        time.sleep(61)


    # Read image bytes
    try:
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
    except Exception as e:
        print(f"⚠️ Error reading image {img_path}: {e}")
        continue  # Skip to the next item if image read fails

    # Fill prompt
    meta_str = json.dumps(extract_metadata(item), indent=2)
    prompt = PROMPT.format(metadata=meta_str)

    print(i, f"Processing: {img_path}")

    # Wait until internet is available
    while not is_connected():
        print("🌐 Internet seems down. Waiting for it to come back...")
        # print("   (Press Enter to exit manually)")
        try:
            time.sleep(15)
        except KeyboardInterrupt:
            input("↩️  Exiting. Press Enter.")
            exit(1)

    try:
        # Send to Gemini 1.5‑flash
        response = client.models.generate_content(
            # model='gemini-1.5-flash',
            model='gemini-2.0-flash',
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                prompt
            ]
        )
    except Exception as e:
        print(f"⚠️ Error during API call for image {img_path}: {e}")
        continue  # Skip to next item if Gemini fails

    # with open('output.txt', 'w') as file:
    #     file.write(response.text)

    try:
        cleaned_json = clean_json_string(response.text)
        print("Got response (clean text)", cleaned_json)
        qa_pairs = json.loads(cleaned_json)
    except json.JSONDecodeError as JSONDE:
        print("Error decoding JSON", JSONDE)
        qa_pairs = []

    for qa in qa_pairs:
        vqa_data.append({
            "image_id":       item["main_image_id"],
            "image_path":     img_path,
            "other_image_id": item.get("other_image_id", []),
            "question":       qa.get("question"),
            "answer":         qa.get("answer")
        })

# 9) Write out timestamped JSON
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = f"{LISTINGS_FILE.removesuffix('.json.gz')} {ts} curated vqa data.json"
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(vqa_data, f, indent=2)

print(f"Saved {len(vqa_data)} entries to {out_file}")
