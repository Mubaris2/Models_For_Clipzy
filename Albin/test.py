import os
import cv2
import base64
import json
from io import BytesIO
from PIL import Image
from collections import Counter
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("❌ API_KEY not found in .env file")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-flash")


def extract_frames(video_path, every_n_frames=80):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n_frames == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames


def encode_image(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def classify_frame(frame):
    b64_image = encode_image(frame)
    response = model.generate_content([
        "Classify this frame into high-level categories like [Sports, Animals, Food, Cooking, Technology, Nature, People, Car]. Respond ONLY with a JSON list of labels. No extra text, no markdown.",
        {"mime_type": "image/jpeg", "data": b64_image}
    ])
    return response.text.strip()


def aggregate_labels(frame_labels):
    all_labels = []
    for lbl in frame_labels:
        # 🔹 Strip code fences if present
        cleaned_lbl = lbl.strip().replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(cleaned_lbl)  # Try strict JSON parse
            all_labels.extend(parsed)
        except:
            # fallback: crude parsing
            cleaned = cleaned_lbl.replace("[", "").replace("]", "").replace('"', "")
            all_labels.extend([x.strip() for x in cleaned.split(",")])

    counts = Counter([x for x in all_labels if x])
    return counts.most_common()



if __name__ == "__main__":
    video_path =  r"E:\ML_CLIPZY\Albin\codes\myvideo.mp4" #replace with video location
  
    frames = extract_frames(video_path, every_n_frames=60)  
    print(f"Extracted {len(frames)} frames")

    
    frame_labels = [classify_frame(f) for f in frames[:5]]  
    print("Raw Frame Labels:", frame_labels)

    final_labels = aggregate_labels(frame_labels)
    print("\n✅ Final Video Labels:", final_labels)
