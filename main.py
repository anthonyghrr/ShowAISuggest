import pickle
from thefuzz import process
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def load_embeddings(filepath):
    with open(filepath, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings

def match_shows(user_input, show_list):
    matching_shows = []
    for show in user_input:
        match = process.extractOne(show, show_list)
        if match is not None:
            match_show, match_score = match
            if match_score >= 70:  
                matching_shows.append(match_show)
    return matching_shows

def calculate_average_vector(selected_shows, embeddings):
    vectors = []
    for show in selected_shows:
        vectors.append(embeddings.get(show))
    avg_vector = np.mean(vectors, axis=0)
    return avg_vector  

def find_similar_shows(avg_vector, embeddings, exclude_shows):
    results = []
    normalized_avg_vector = normalize([avg_vector])[0]
    
    for title, vector in embeddings.items():
        if title not in exclude_shows:
            normalized_vector = normalize([vector])[0]
            similarity = cosine_similarity([normalized_avg_vector], [normalized_vector])[0][0]
            results.append((title, similarity))
    
    return sorted(results, key=lambda x: x[1], reverse=True)[:5]

def get_shows_and_confirm(embeddings):
    while True:
        user_input = input("Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show\n")
        shows_list = user_input.split(",")
        user_shows = [show.strip() for show in shows_list]
        matched_shows = match_shows(user_shows, list(embeddings.keys()))

        if matched_shows:
            print(f"Do you mean {', '.join(matched_shows)}? (y/n)")
            if input().lower() == 'y':
                print(f"Great! Proceeding with the following shows: {', '.join(matched_shows)}")
                avg_vector = calculate_average_vector(matched_shows, embeddings)
                recommended_shows = find_similar_shows(avg_vector, embeddings, matched_shows)
                print("We recommend the following shows based on your preferences:")
                for show, similarity in recommended_shows:
                    print(f"{show} ({similarity * 100:.0f}%)")

                break
            else:
                print("Please re-enter the shows with correct spelling.")
        else:
            print("Sorry, no matches found. Please check your spelling and try again.")

if __name__ == "__main__":
    # Load embeddings with the updated path
    embeddings = load_embeddings("/Users/anthonyghandour/Desktop/ShowAISuggest/src/data/embeddings.pkl")
    get_shows_and_confirm(embeddings)
