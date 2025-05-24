import pandas as pd
import numpy as np
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, FollowupAction
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List
import re

class PreFormValidation(Action):
  def name(self) -> Text:
    return "action_pre_form_validation"

  def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
    invalid_slots = []
    budget = budget_validation(tracker)
    if budget and isinstance(budget, dict) and budget["budget"] is None:
      invalid_slots.append("Budget")
    screen_size = screen_validation(tracker)
    if screen_size and isinstance(screen_size, dict) and screen_size["screen_size"] is None:
      invalid_slots.append("Screen Size")
    storage_size = storage_validation(tracker)
    if storage_size and isinstance(storage_size, dict) and storage_size["storage_size"] is None:
      invalid_slots.append("Storage Size")
    ram_size = ram_validation(tracker)
    if ram_size and isinstance(ram_size, dict) and ram_size["ram"] is None:
      invalid_slots.append("RAM")
      
      return []

class ValidateLaptopSpecsForm(FormValidationAction):
  def name(self) -> Text:
    return "validate_laptop_specs_form"

  def validate_budget(self, slot_value, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
    values = re.findall(r"\d+", str(slot_value).replace(",", ""))

    if tracker.get_slot("ram") is not None:
      return [SlotSet("ram", None)]

    if values:
      budget = list(map(int, values))
      if any(b < 150 or b > 5000 for b in budget):
        dispatcher.utter_message("Please enter a valid budget between $150 and $5000.")
        return{"budget": None}
      else:
        return{"budget": budget}
    else:
      dispatcher.utter_message("I didn't catch that.")
      return{"budget": None}

  def validate_storage_size(self, slot_value, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
    extracted = re.findall(r"(\d+\s*(?:GB|TB))", str(slot_value), re.IGNORECASE)

    if tracker.get_slot("ram") is not None:
      return [SlotSet("ram", None)]

    if extracted:
      sizes = [size.replace(" ", "").upper() for size in extracted]
      valid = ["128GB", "256GB", "512GB", "1TB", "2TB"]

      if all(size in valid for size in sizes) and len(sizes) <= 2:
          return {"storage_size": sizes} 
      else:
        dispatcher.utter_message("Please enter a valid storage size like 512GB, 1TB, 512GB + 1TB.")
        return {"storage_size": None}
    else:
      dispatcher.utter_message("I didn't catch that.")
      return{"storage_size": None}

  def validate_ram(self, slot_value, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
    value = re.search(r"\b(\d+)\s*gb\b", str(slot_value).strip(), re.IGNORECASE)
    if value:
      ram_size = value.group(1) + "GB"
      sizes = ["2GB","3GB","4GB","6GB","8GB","12GB","16GB","20GB","24GB","32GB","40GB","48GB","64GB", "80GB"]
      if ram_size in sizes:
        return{"ram": ram_size}
      else:
        dispatcher.utter_message("Please enter a valid ram size.")
        return{"ram": None}
    else:
      dispatcher.utter_message("I didn't catch that.")
      return{"ram": None}

  def validate_screen_size(self, slot_value, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
    value = re.search(r"\d+(\.\d+)?", str(slot_value))
    if value:
      size = float(value.group())
      if 11.0 <= size <= 20.0:
        return{"screen_size": size}
      else:
        dispatcher.utter_message("Please enter a valid screen size (e.g., 13.3, 15.6, or 17 inches).")
        return{"screen_size": None}
    else:
      dispatcher.utter_message("I didn't catch that.")
      return{"screen_size": None}

def budget_validation(tracker):
  budget = tracker.get_slot("budget")
  if isinstance(budget, list) and all (isinstance(b, int) for b in budget):
    budget = sorted(budget)
    if any(b < 150 or b > 5000 for b in budget):
      return{"budget": None}
    else:
      return {"budget": budget}
  if isinstance(budget, int):
    if 150 <= budget <= 5000:
        return {"budget": [budget]}
    else:
        return {"budget": None}
  return {"budget": None}

def ram_validation(tracker):
  slot_value = tracker.get_slot("ram")
  if not slot_value:
    return{"ram": None}
  slot_value = str(slot_value).strip().lower()
  value = re.search(r"\b(\d{1,2})\s*gb\b", slot_value, re.IGNORECASE)
  sizes = ["2GB","3GB","4GB","6GB","8GB","12GB","16GB","20GB","24GB","32GB","40GB","48GB","64GB"]
  if value: 
    ram = value.group(1) + "GB"
    if ram in sizes:
      return{"ram": ram}
    else:
      return{"ram": None}

def screen_validation(tracker):
  value = tracker.get_slot("screen_size")
  screen = re.search(r"\d+(\.\d+)?", str(value))
  if screen:
    size = float(screen.group())
    if 11.0 <= size <= 20.0:
      return{"screen_size": size}
    else:
      return{"screen_size": None}

def storage_validation(tracker):
  storage_size = tracker.get_slot("storage")
  sizes = ["128GB","256GB","512GB","1TB","2TB"]
  if storage_size:
    if storage_size in sizes:
      return {"storage_size": storage_size}
    else:
      return {"storage_size": None}

class ActionNoPreference (Action):
  def name(self) -> Text:
    return "action_no_preference" 
  
  def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
    last_bot_message = None
    for event in reversed(tracker.events):
      if event.get("event") == "bot":
        last_bot_message = event.get("text")
        break

    slot_to_set = None
    if last_bot_message:
      if "budget" in last_bot_message:
        slot_to_set = "budget"
      elif "gaming" in last_bot_message:
        slot_to_set = "laptop_type"
      elif "type of storage" in last_bot_message:
        slot_to_set = "storage_type"
      elif "much storage" in last_bot_message:
        slot_to_set = "storage_size"
      elif "RAM" in last_bot_message:
        slot_to_set = "ram"
      elif "processor" in last_bot_message:
        slot_to_set = "processor"
      elif "GPU" in last_bot_message:
        slot_to_set = "gpu"
      elif "screen size" in last_bot_message:
        slot_to_set = "screen_size"
      elif "operating system" in last_bot_message:
        slot_to_set = "os"
      elif "brand" in last_bot_message:
        slot_to_set = "brand_name"
      elif "resolution" in last_bot_message:
        slot_to_set = "resolution"
      elif "feature" in last_bot_message:
        slot_to_set = "feature"
      
      if slot_to_set:
        dispatcher.utter_message(text="Got it! I'll set this preference to neutral.")
        return[SlotSet(slot_to_set,"neutral"), FollowupAction("laptop_specs_form")]

class ActionAdjustPreference (Action):
  def name(self) -> Text:
    return "action_adjust_preference"
  
  def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
    dispatcher.utter_message(text= "Got it! I'll adjust your preferences as needed!")
    return [FollowupAction("action_summarize")]

class ActionSummarize (Action):
  def name(self):
    return "action_summarize"

  def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
      slot_names = [
          "laptop_type", "processor", "ram", "storage_size", 
          "storage_type", "gpu", "screen_size", "budget",
          "resolution", "brand_name", "os"
      ]
      
      user_preferences = {}

      for slot in slot_names:
          slot_value = tracker.get_slot(slot)
          if slot_value:  
              user_preferences[slot] = slot_value
      
      if len(user_preferences) == 11:
          summary_lines = [
              f"- {slot.replace('_', ' ').title()}: {value}" 
              for slot, value in user_preferences.items()
          ]
          summary = "\n".join(summary_lines)  
          dispatcher.utter_message(text=f"Here's what I have so far:\n\n{summary}\n\nWould you like to change anything or specify any more preferences?")
          return [FollowupAction("action_listen")]
      else:
          dispatcher.utter_message(text="There are some missing preferences.")

class ActionTriggerScores (Action):
  def name(self) -> Text:
    return "action_trigger_scores"
  
  def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
    dispatcher.utter_message(json_message={"data":{"trigger_scores_ui": True}})
    return[]

class ActionClarifyComponent(Action):
  def name(self) -> Text:
    return "action_explain_component"

  def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
    entities = tracker.latest_message.get("entities", [])
    component = None

    for entity in entities:
      if entity["entity"] in ["component", "storage_type", "feature"]:
        component = entity["value"]
        break

    if not component:
      dispatcher.utter_message(text="I couldn't recognize the component you're asking about. Could you please rephrase?")
      return []
    else: 
      component = component.lower()
      file_path = r"C:\Users\DellPc\Desktop\Senior\laptop_components.csv"
      file = pd.read_csv(file_path, encoding="ISO-8859-1")

      component_info = file[file.iloc[:, 0].str.lower() == component]

      if not component_info.empty:
        row = component_info.iloc[0]
        definition = row.iloc[1]
        examples = row.iloc[2]

        dispatcher.utter_message(text=f"{definition}\nHere are some examples: {examples}")
      else: 
        dispatcher.utter_message(text=f"{component} is: I couldn't find any information on that component. Try asking about another one!")

    requested_slot = tracker.active_loop.get("requested_slot")

    if requested_slot:
      slot_prompt_map = {
                "budget": "What is your budget for this laptop?",
                "laptop_type": "For what purposes do you wish to use this laptop for?",
                "storage_type": "What type of storage do you prefer for your laptop? (e.g., SSD, HDD, Dual Storage, Hybrid Drive)",
                "storage_size": "How much storage do you need? (e.g., 256GB, 512GB, or 1TB)",
                "ram": "How much RAM would you like? 8GB, 16GB, or more?",
                "processor": "What type of processor do you prefer? (e.g., Intel Core i7, AMD Ryzen 9, etc.)",
                "gpu": "What GPU would you like your laptop to have? entry-level, mid-range, or high-end? You can choose to have no gpu.",
                "screen_size": "What screen size do you prefer? (e.g., 13-inch, 15-inch, etc.)",
                "resolution": "What screen resolution would you like your screen to have? HD, Ultra HD, or Retina Display?",
                "os": "Do you have a preference for an operating system?",
                "brand_name": "Would you like a laptop suggestion from a specific brand?"
                }
          
      prompt = slot_prompt_map.get(requested_slot, "Let's continue with your laptop preferences.")
      dispatcher.utter_message(text=f"Now, let's get back to our conversation: {prompt}")
      return [FollowupAction("action_listen")]
    else:
      dispatcher.utter_message(text="Let's continue with your laptop preferences.")
      return [FollowupAction("action_listen")]

df = pd.read_csv(r"C:\Users\DellPc\Downloads\bitmasked_laptops_with_updated_storage_config(1).csv", encoding="ISO-8859-1", dtype={"RAM Bucket": str})
sentiment_df = pd.read_csv(r"C:\Users\DellPc\Downloads\All_laptops_with_avg_sentiment.csv", encoding="ISO-8859-1")
reviews_df = pd.read_csv(r"C:\Users\DellPc\Downloads\All specifications.csv", encoding="ISO-8859-1")

def normalize_os(text):
  if not text: return None
  text = text.lower().replace(" ", "")
  if "windows11" in text: return "Windows 11"
  elif "windows10" in text: return "Windows 10"
  elif "windows8" in text: return "Windows 8"
  elif "windows7" in text: return "Windows 7"
  elif text == "windows": return "windows_family"
  elif "chrome" in text: return "Chrome OS"
  elif "mac" in text: return "macOS"
  return text.title()

def bucket_price(p):
  if p < 800: return "<800"
  elif p < 1000: return "800-1000"
  elif p < 1200: return "1000-1200"
  elif p < 1600: return "1200-1600"
  elif p < 2000: return "1600-2000"
  elif p < 2500: return "2000-2500"
  elif p < 3000: return "2500-3000"
  elif p < 4000: return "3500-4000"
  else: return "+4000"

user_resolution_map = {
    "hd": "Low Res",
    "full hd": "Medium Res",
    "ultra hd": "Medium Res",
    "2k": "Medium Res",
    "quad hd": "Medium Res",
    "qhd": "Medium Res",
    "4k": "High Res",
    "uhd":"High Res"
}

def extract_float(text):
  try: return float(re.search(r"\d+(\.\d+)?", str(text)).group())
  except: return None

def parse_ram(text):
  try: return int(re.search(r"\d+", str(text)).group())
  except: return None

def parse_price(text):
  try: return int(re.search(r"\d+", str(text).replace(",", "")).group())
  except: return None

def map_screen_size_to_bucket(size):
  if size < 13:
      return "<13"
  elif size < 15:
      return "13-15"
  elif size <= 17:
      return "15-17"
  else:
      return ">17"

def apply_screen_size_filter(df, size_str):
  size = extract_float(size_str)
  if size:
      bucket = map_screen_size_to_bucket(size)
      return df[df["screen_size_bucket"] == bucket]
  return df

def parse_storage_size(size_str):
  match = re.search(r'(\d+(\.\d+)?)(tb|gb)', size_str.lower().replace(" ", ""))
  if match:
      size = float(match.group(1))
      unit = match.group(3)
      return size * 1024 if unit == "tb" else size
  return None

def apply_storage_filter(df, storage_type, storage_list):
  df = df.copy()

  def parse_gb(text):
    if isinstance(text, str):
      match = re.search(r"(\d+)", text.lower())
      return int(match.group()) if match else None
    return int(text) if pd.notnull(text) else None

  parsed_sizes = [parse_gb(s) for s in storage_list if parse_gb(s) is not None]
  if not parsed_sizes:
    return df 

  storage_type = (storage_type or "").lower().strip()

  if storage_type == "ssd only":
    return df[(df["ssd_size_gb"].fillna(0) >= max(parsed_sizes)) & (df["hdd_size_gb"].fillna(0) == 0)]

  elif storage_type == "hdd only":
    return df[(df["hdd_size_gb"].fillna(0) >= max(parsed_sizes)) & (df["ssd_size_gb"].fillna(0) == 0)]

  elif storage_type == "dual":
    if len(parsed_sizes) == 2:
      s1, s2 = parsed_sizes
      return df[
              ((df["ssd_size_gb"].fillna(0) >= s1) & (df["hdd_size_gb"].fillna(0) >= s2)) |
              ((df["ssd_size_gb"].fillna(0) >= s2) & (df["hdd_size_gb"].fillna(0) >= s1))
              ]
    elif len(parsed_sizes) == 1:
      s = parsed_sizes[0]
      return df[
              ((df["ssd_size_gb"].fillna(0) >= s * 2) & (df["hdd_size_gb"].fillna(0) == 0)) |
              ((df["hdd_size_gb"].fillna(0) >= s * 2) & (df["ssd_size_gb"].fillna(0) == 0))
              ]

  elif storage_type == "dual ssd":
    if len(parsed_sizes) == 2:
      s1, s2 = parsed_sizes
      return df[
              (df["ssd_size_gb"].fillna(0) >= s1 + s2) & (df["hdd_size_gb"].fillna(0) == 0)
              ]
    elif len(parsed_sizes) == 1:
      s = parsed_sizes[0]
      return df[
              (df["ssd_size_gb"].fillna(0) >= s * 2) & (df["hdd_size_gb"].fillna(0) == 0)
              ]

  elif storage_type == "dual hdd":
    if len(parsed_sizes) == 2:
      s1, s2 = parsed_sizes
      return df[
              (df["hdd_size_gb"].fillna(0) >= s1 + s2) & (df["ssd_size_gb"].fillna(0) == 0)
              ]
    elif len(parsed_sizes) == 1:
      s = parsed_sizes[0]
      return df[
              (df["hdd_size_gb"].fillna(0) >= s * 2) & (df["ssd_size_gb"].fillna(0) == 0)
              ]

  elif storage_list:
    return df[
            (df["ssd_size_gb"].fillna(0) >= max(parsed_sizes)) |
            (df["hdd_size_gb"].fillna(0) >= max(parsed_sizes))
            ]

  return df

gpu_tier_order = {
    "No GPU": 1, "Integrated GPU": 2, "Entry-Level GPU": 3,
    "Mid-Range GPU": 4, "High-End GPU": 5
}
screen_res_order = {"Low Res": 1, "Medium Res": 2, "High Res": 3}



filter_order = [
    "brand", "laptop_type", "cpu", "gpu", "screen_size", "resolution",
    "budget", "ram", "storage_type", "storage_size", "os"
]

relaxation_priority = [
    "brand", "os", "storage_type", "storage_size",
    "resolution", "screen_size", "ram", "gpu"
]

def filter_laptops(user_input, df, dispatcher):
  relaxed = set()
  for i in range(len(relaxation_priority) + 1):
    current_df = df.copy()
    relaxed_so_far = set()

    for key in filter_order:
      if key in relaxed or key in relaxed_so_far or user_input.get(key) in [None, "neutral"]:
        continue

      value = user_input.get(key)

      if key == "brand":
        current_df = current_df[current_df["brand"].str.lower() == value.lower()]
      elif key == "laptop_type":
        current_df = current_df[current_df["Laptop Type"].str.lower() == value.lower()]
      elif key == "cpu":
        current_df = current_df[current_df["CPU"].str.lower().str.contains(value.lower())]
      elif key == "gpu":
        mapping = {
                  "no gpu": "No GPU", "integrated": "Integrated GPU",
                  "entry level": "Entry-Level GPU", "mid range": "Mid-Range GPU",
                  "high end": "High-End GPU"
                }
        tier = mapping.get(value.lower().replace("-", " "), None)
        if tier:
          tier_rank = gpu_tier_order.get(tier)
        if tier_rank:
          current_df = current_df[current_df["GPU Tier"].map(gpu_tier_order.get) >= tier_rank]
      elif key == "screen_size":
        current_df = apply_screen_size_filter(current_df, value)
      elif key == "resolution":
        val = screen_res_order.get(user_resolution_map.get(value.lower()))
        if val:
          current_df = current_df[current_df["Screen Resolution Bucket"].map(screen_res_order.get) >= val]
      elif key == "budget":
          amount = parse_price(value)
          if amount:
            current_df = current_df[current_df["price"] <= amount]
      elif key == "ram":
        ram_val = parse_ram(value)
        if ram_val:
          current_df["RAM (GB)"] = pd.to_numeric(current_df["RAM"].str.extract(r"(\d+)")[0], errors='coerce')
          current_df = current_df[current_df["RAM (GB)"] >= ram_val]
      elif key in ["storage_size", "storage_type"]:
        current_df = apply_storage_filter(current_df, user_input.get("storage_type"), user_input.get("storage_size"))
      elif key == "os":
        norm_os = normalize_os(value)
        if norm_os == "windows_family":
          current_df = current_df[current_df["Normalized OS"].str.lower().str.contains("windows")]
        else:
          current_df = current_df[current_df["Normalized OS"].str.lower() == norm_os.lower()]

        print(f"After {key}: {len(current_df)}")
        if current_df.empty and key in relaxation_priority[i:]:
          relaxed_so_far.add(key)
          print(f"Relaxing: {key}")
          break

    relaxed.update(relaxed_so_far)

    if not current_df.empty:
      relaxed.update(relaxed_so_far)
      if(len(current_df) > 1):
        dispatcher.utter_message(f"\nBased on your preferences, {len(current_df)} laptop matches were found")
      else:
        dispatcher.utter_message(f"\nBased on your preferences, one laptop match was found")
      return current_df.reset_index(drop=True), list(relaxed)
    
    return pd.DataFrame(), list(relaxed)

def rank_with_topsis(df_filtered, user_priorities, reviews_df, dispatcher):
  merged = pd.merge(df_filtered, sentiment_df, on="id", how="inner")
  if merged.empty:
    return pd.DataFrame()

  merged = pd.merge(merged, reviews_df[['id', 'num_reviews', 'avg_helpful_votes', 'average_rating', 'rating_number']],
                    on="id", how="left")

  merged['num_reviews'] = merged['num_reviews'].fillna(1)
  merged['avg_helpful_votes'] = merged['avg_helpful_votes'].fillna(0)
  merged['average_rating'] = merged['average_rating'].fillna(3)  
  merged['rating_number'] = merged['rating_number'].fillna(1)
  
  all_features = [
      "Performance", "Build quality", "Battery life", "Display",
      "Gaming", "Graphics", "Sound", "Fans", "Cooling system", "Weight", "Price"
  ]
  
  weights_dict = {feat: user_priorities.get(feat, 0) for feat in all_features}
  active_features = [f for f in all_features if weights_dict[f] > 0]
  
  if not active_features:
    return merged

  X_adjusted = np.zeros((len(merged), len(active_features)))

  def calculate_confidence_factor(row, weights=(0.5, 0.2, 0.3)):
    w_review, w_helpful, w_rating = weights

    review_factor = np.log1p(row['num_reviews']) / np.log1p(100)
    review_factor = min(review_factor, 1.0)

    helpful_factor = min(row['avg_helpful_votes'] / 5.0, 1.0)

    rating_factor = np.log1p(row['rating_number']) / np.log1p(100)
    rating_factor = min(rating_factor, 1.0)

    return w_review * review_factor + w_helpful * helpful_factor + w_rating * rating_factor

  merged['confidence_factor'] = merged.apply(calculate_confidence_factor, axis=1)

  for i, feat in enumerate(active_features):
    for j, (_, row) in enumerate(merged.iterrows()):
      raw_sentiment = row[feat]
      confidence = row['confidence_factor']
      X_adjusted[j, i] = raw_sentiment * confidence
      
  norm_X = X_adjusted + 1
  
  raw_weights = np.array([weights_dict[f] for f in active_features], dtype=float)
  w = raw_weights / np.sum(raw_weights) if np.sum(raw_weights) > 0 else np.ones_like(raw_weights) / len(raw_weights)

  print(f"\nFeature weights:")
  for i, feat in enumerate(active_features):
    print(f"  - {feat}: {w[i]:.2f}")

  V = norm_X * w

  ideal = np.max(V, axis=0)     
  anti_ideal = np.min(V, axis=0)
  
  d_plus = np.sqrt(np.sum((V - ideal)**2, axis=1))    
  d_minus = np.sqrt(np.sum((V - anti_ideal)**2, axis=1))  

  denominator = d_plus + d_minus
  scores = np.where(denominator > 0, d_minus / denominator, 0)

  merged["topsis_score"] = scores

  for i, feat in enumerate(active_features):
    merged[f"adjusted_{feat}"] = X_adjusted[:, i]
    merged[f"weighted_{feat}"] = (X_adjusted[:, i] + 1) * w[i]

  print("\nFull Ranked Laptops (Top 10):")
  top_df = merged[["brand", "laptop_title_x", "price", "topsis_score", "num_reviews", "confidence_factor"]].sort_values("topsis_score", ascending=False).head(10)
  print(top_df[["brand", "laptop_title_x", "price", "topsis_score", "num_reviews", "confidence_factor"]])

  print("\nTop 3 Recommendations:")
  top3 = merged.sort_values("topsis_score", ascending=False).head(3)
  for i, row in top3.iterrows():
    print(f"{i+1}. {row['brand']} {row['laptop_title_x']} - ${row['price']} (Score: {row['topsis_score']:.4f})")
    print(f"   Based on {int(row['num_reviews'])} reviews with {row['confidence_factor']:.2f} confidence factor")

    if 'average_rating' in row and row['average_rating'] > 0:
      print(f"   Average rating: {row['average_rating']:.1f}/5 from {int(row['rating_number'])} ratings")

    contributions = [(f, row[f"weighted_{f}"], row[f], row[f"adjusted_{f}"])
                      for f in active_features]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top factors:")
    for f, weighted, raw, adjusted in contributions[:3]:
      sentiment_level = "positive" if raw > 0.3 else "negative" if raw < -0.3 else "neutral"
      confidence_impact = "" if abs(raw - adjusted) < 0.05 else f" (adjusted from {raw:.2f})"
      print(f"     - {f}: {sentiment_level} ({adjusted:.2f}{confidence_impact})")

  return merged.sort_values("topsis_score", ascending=False).head(3)

class ActionProvideRecommendation(Action):
  def name(self) -> Text:
    return "action_provide_recommendation"

  def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
    slots = tracker.slots
    user_input = {
          "brand": slots.get("brand_name"),
          "laptop_type": slots.get("laptop_type"),
          "cpu": slots.get("processor"),
          "gpu": slots.get("gpu"),
          "os": slots.get("os"),
          "screen_size": slots.get("screen_size"),
          "resolution": slots.get("resolution"),
          "ram": slots.get("ram"),
          "budget": slots.get("budget"),
          "storage_type": slots.get("storage_type"),
          "storage_size": slots.get("storage_size"),
          "priorities": tracker.latest_message.get("metadata", {}).get("priorities", {})
      }

    df_filtered, relaxed_filters = filter_laptops(user_input, df, dispatcher)

    if df_filtered.empty:
      dispatcher.utter_message("Unfortunately, no laptops matched your preferences.")
      return []

    if relaxed_filters: 
      dispatcher.utter_message(f"Note: Some preferences were relaxed to find the best results. The following are the specs that got relaxed to reach these laptop matches: {list(relaxed_filters)}")
      
    priority_list = user_input["priorities"]
    priority_dict = {item["priority"]: int(item["value"]) for item in priority_list}
    top_laptops = rank_with_topsis(df_filtered, priority_dict, reviews_df, dispatcher)

    if top_laptops.empty: 
      dispatcher.utter_message("Unfortunately, no laptops matched your preferences after ranking.")
      return []
    
    response = ""
    i = 1

    for idx, row in top_laptops.iterrows():
      title = row.get("laptop_title_x", "Unknown model")
      brand = row.get("brand", "")
      price = row.get("price", "N/A")
      screen = row.get("Standing screen display size", "Unknown")
      ram = row.get("RAM", "Unknown")
      storage = row.get("Hard Drive", "Unknown")
      processor = row.get("CPU", "Unknown")
      gpu = row.get("Graphics Coprocessor", "Unknown")
      resolution = row.get("Screen Resolution", "Unknown")
      os = row.get("Operating System", "Unknown")

      response += f"Laptop {i}:\n"
      response += f"{title}\n"
      response += f"Price: ${price}\n"
      response += f"Processor: {processor}\n"
      response += f"RAM: {ram}\nStorage: {storage}\n"
      response += f"Screen: {screen}\nResolution: {resolution}\n"
      response += f"GPU: {gpu}\nOS: {os}\n\n\n"

      i += 1

    dispatcher.utter_message(text=response)
    return []
