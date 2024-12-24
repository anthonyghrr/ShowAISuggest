from openai import OpenAI
import pandas as pd
import pickle
import os

# Initialize OpenAI client
client = OpenAI(api_key="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")

# Load CSV
df = pd.read_csv("/Users/anthonyghandour/Desktop/ShowAISuggest/data/tv_shows.csv")

# Generate embeddings
embeddings = {}
for row in df.itertuples(index=False):
    title = row.Title
    description = row.Description
    try:
        response = client.embeddings.create(
            input=description,
            model="text-embedding-3-small"
        )
        embeddings[title] = response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding for {title}: {e}")

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Save embeddings
with open(os.path.join(output_dir, "embeddings.pkl"), "wb") as f:
    pickle.dump(embeddings, f)
