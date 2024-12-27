import pickle
from dotenv import load_dotenv
import requests
import numpy as np
import random
import time
import os
from thefuzz import process
import json
from colorama import init, Fore, Style
from requests import HTTPError
import re

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()
LIGHTX_API_KEY = os.getenv('LIGHTX_API_KEY')
EMBEDDINGS_PATH = "/Users/anthonyghandour/Desktop/ShowAISuggest/src/data/embeddings.pkl"

def load_embeddings(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def fuzzy_match_shows(user_input_shows, all_show_titles):
    matched = []
    for show in user_input_shows:
        best_title, best_score = process.extractOne(show, all_show_titles)
        if best_score >= 70:
            matched.append(best_title)
    return matched

def cosine_similarity(vec_a, vec_b):
    dot_prod = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_prod / (norm_a * norm_b)

# Generate a random show name based on description text
def create_random_show_name_from_description(description):
    if not description:
        return "Untitled Saga"

    ignore_words = {"of", "the", "and", "a", "to", "in", "on", "for", "from", "by", "with", "as", "at", "this", "that"}

    words = re.findall(r'\b\w{3,}\b', description.lower())  
    filtered_words = [word.capitalize() for word in words if word not in ignore_words]
    
    if not filtered_words:
        return "Mysterious Saga"
    num_words = random.choice([1, 2, 3])
    name_words = random.sample(filtered_words, k=min(num_words, len(filtered_words)))
    return " ".join(name_words)

def request_image_generation(prompt: str):
    url = 'https://api.lightxeditor.com/external/api/v1/text2image'
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': LIGHTX_API_KEY
    }
    data = {"textPrompt": str(prompt)}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["body"]["orderId"]
    else:
        print(f"{Fore.RED}Request failed with status code: {response.status_code}")
        print(response.json())
        return None

def fetch_image_status(order_id: str) -> str:
    WAIT_TIME = 3
    MAX_TRIES = 5

    url = 'https://api.lightxeditor.com/external/api/v1/order-status'
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': LIGHTX_API_KEY
    }
    payload = {"orderId": order_id}

    for _ in range(MAX_TRIES):
        time.sleep(WAIT_TIME)
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            status = response.json()["body"]["status"]
            if status == 'init':
                continue
            elif status == 'failed':
                raise Exception("Image Generation Failed")
            else:
                return response.json()["body"]['output']
        else:
            raise HTTPError(f"Error: Received status code {response.status_code}")

def download_and_open_image(image_url: str, filename: str):
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        os.system(f'open {filename}')
    else:
        print(f"{Fore.RED}Failed to download image from {image_url}")

def generate_image(prompt, show_name):
    order_id = request_image_generation(prompt.strip())
    if order_id:
        image_url = fetch_image_status(order_id)
        if image_url:
            filename = f"{show_name.replace(' ', '_')}.jpg"
            download_and_open_image(image_url, filename)
            return filename
    return None

def main():
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    all_show_titles = list(embeddings.keys())

    while True:
        user_input = input(
            f"{Fore.CYAN}Which TV shows did you really like watching? "
            "Separate them by a comma. Make sure to enter more than 1 show\n"
        )
        user_shows_raw = [s.strip() for s in user_input.split(",") if s.strip()]

        if len(user_shows_raw) < 2:
            print(f"{Fore.RED}Please enter at least 2 shows!\n")
            continue

        matched_shows = fuzzy_match_shows(user_shows_raw, all_show_titles)
        if not matched_shows:
            print(f"{Fore.RED}No matches found. Try again.\n")
            continue

        print(f"{Fore.YELLOW}Making sure, do you mean {', '.join(matched_shows)}? (y/n)")
        confirm = input().lower().strip()
        if confirm == 'y':
            break
        else:
            print(f"{Fore.CYAN}Ok let's try again.\n")

    print(f"{Fore.GREEN}Great! Generating recommendations now...")

    user_vectors = [embeddings[s] for s in matched_shows]
    avg_vector = np.mean(user_vectors, axis=0)

    rec_list = []
    for show_title, vec in embeddings.items():
        if show_title in matched_shows:
            continue
        sim = cosine_similarity(avg_vector, vec)
        rec_list.append((show_title, sim))

    rec_list.sort(key=lambda x: x[1], reverse=True)

    top_five = rec_list[:5]

    recommended_shows = []
    for (title, sim) in top_five:
        pct = (sim + 1) * 50
        pct = max(0, min(100, pct))
        recommended_shows.append((title, pct))

    print(f"{Fore.GREEN}\nHere are the TV shows that I think you would love:")
    for title, pct in recommended_shows:
        print(f"{Fore.GREEN}{title} ({pct:.0f}%)")
    print()

    show1_name = create_random_show_name_from_description(f"A thrilling saga inspired by {', '.join(matched_shows)}.")
    show1_desc = f"A thrilling saga inspired by {', '.join(matched_shows)}."
    show1_prompt = f"A scene from {show1_name}, inspired by {', '.join(matched_shows)}. The show name is {show1_name}."

    show2_name = create_random_show_name_from_description(f"A masterpiece blending the themes of {', '.join(title for title, _ in recommended_shows)}.")
    show2_desc = f"A masterpiece blending the themes of {', '.join(title for title, _ in recommended_shows)}."
    show2_prompt = f"A scene from {show2_name}, inspired by {', '.join(title for title, _ in recommended_shows)}. The show name is {show2_name}."

    try:
        print(
            f"{Fore.CYAN}\nI have also created just for you two custom shows which I think you would love.\n\n"
            f"{Style.RESET_ALL}Show #1 is based on the fact that you loved the input shows that you gave me. "
            f"Its name is {Fore.YELLOW}{Style.BRIGHT}{show1_name}{Style.RESET_ALL} and it is about {show1_desc}.\n"
        )
        time.sleep(2)
        generate_image(show1_prompt, show1_name)

        print(
            f"\nShow #2 is based on the shows that I recommended for you. "
            f"Its name is {Fore.YELLOW}{Style.BRIGHT}{show2_name}{Style.RESET_ALL} and it is about {show2_desc}.\n"
            f"{Fore.CYAN}Hope you like them!{Style.RESET_ALL}\n"
        )
        time.sleep(2)
        generate_image(show2_prompt, show2_name)

    except Exception as e:
        print(f"{Fore.RED}Image generation failed: {e}")

if __name__ == "__main__":
    main()