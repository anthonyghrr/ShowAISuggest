import pickle
from thefuzz import process

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
            if match_score >= 80: 
                matching_shows.append(match_show)
    return matching_shows

def get_shows_and_confirm(embeddings):
    while True:
        user_input = input("Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show\n")
        user_shows = [s.strip() for s in user_input.split(",")]
        matched_shows = match_shows(user_shows, list(embeddings.keys()))

        if matched_shows:
            print(f"Do you mean {', '.join(matched_shows)}? (y/n)")
            if input().lower() == 'y':
                print(f"Great! Proceeding with the following shows: {', '.join(matched_shows)}")
                break
            else:
                print("Please re-enter the shows with correct spelling.")
        else:
            print("Sorry, no matches found. Please check your spelling and try again.")

if __name__ == "__main__":
    # Load embeddings with the updated path
    embeddings = load_embeddings("/Users/anthonyghandour/Desktop/ShowAISuggest/src/data/embeddings.pkl")
    get_shows_and_confirm(embeddings)
