import unittest
from unittest.mock import patch
import numpy as np
import pickle
from main import load_embeddings, match_shows, calculate_average_vector, find_similar_shows, get_shows_and_confirm  
from thefuzz import process
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# Mocked version of load_embeddings to simulate data
def mocked_load_embeddings(filepath):
    return {
        "Breaking Bad": np.array([0.1, 0.2, 0.3]),
        "Better Call Saul": np.array([0.1, 0.2, 0.4]),
        "Stranger Things": np.array([0.5, 0.6, 0.7])
    }

class TestShowMatcher(unittest.TestCase):

    # Test for match_shows with valid matches
    def test_match_shows(self):
        show_list = ["Breaking Bad", "Better Call Saul", "Stranger Things"]
        user_input = ["breaking bad", "stranger things"]
        matched_shows = match_shows(user_input, show_list)
        self.assertEqual(len(matched_shows), 2)
        self.assertIn("Breaking Bad", matched_shows)
        self.assertIn("Stranger Things", matched_shows)

    # Test for match_shows with no matches
    def test_match_shows_no_match(self):
        user_input = ["Unknown Show"]
        show_list = ["Breaking Bad", "Stranger Things"]
        matched_shows = match_shows(user_input, show_list)
        self.assertEqual(matched_shows, [])

    # Test for calculate_average_vector
    def test_calculate_average_vector(self):
        embeddings = mocked_load_embeddings("/path/to/embeddings.pkl")
        selected_shows = ["Breaking Bad", "Better Call Saul"]
        avg_vector = calculate_average_vector(selected_shows, embeddings)
        expected_avg = np.mean([embeddings["Breaking Bad"], embeddings["Better Call Saul"]], axis=0)
        np.testing.assert_array_equal(avg_vector, expected_avg)

    # Test for find_similar_shows
    def test_find_similar_shows(self):
        embeddings = mocked_load_embeddings("/path/to/embeddings.pkl")
        avg_vector = np.array([0.15, 0.2, 0.35])
        exclude_shows = ["Breaking Bad"]
        recommended_shows = find_similar_shows(avg_vector, embeddings, exclude_shows)
        self.assertEqual(len(recommended_shows), 2)
        self.assertEqual(recommended_shows[0][0], "Better Call Saul")
        self.assertEqual(recommended_shows[1][0], "Stranger Things")

    # Test for the user input confirmation flow (valid input)
    @patch('builtins.input', side_effect=["Breaking Bad, Better Call Saul", "y"])
    def test_user_confirmation_valid(self, mock_input):
        embeddings = mocked_load_embeddings("/path/to/embeddings.pkl")
        user_input = ["breaking bad", "better call saul"]
        matched_shows = match_shows(user_input, list(embeddings.keys()))
        self.assertEqual(matched_shows, ["Breaking Bad", "Better Call Saul"])

    # Test for the user input confirmation flow (invalid input)
    @patch('builtins.input', side_effect=["Unknown Show", "n", "breaking bad", "y"])
    def test_user_confirmation_invalid(self, mock_input):
        embeddings = mocked_load_embeddings("/path/to/embeddings.pkl")
        user_input = ["Unknown Show"]
        matched_shows = match_shows(user_input, list(embeddings.keys()))
        self.assertEqual(matched_shows, [])

    # Test for loading embeddings with mocked data
    @patch('builtins.open', unittest.mock.mock_open(read_data=pickle.dumps(mocked_load_embeddings("/path/to/embeddings.pkl"))))
    def test_load_embeddings(self):
        embeddings = load_embeddings("/Users/anthonyghandour/Desktop/ShowAISuggest/src/data/embeddings.pkl")
        self.assertEqual(len(embeddings), 3)
        self.assertIn("Breaking Bad", embeddings)
        self.assertEqual(embeddings["Breaking Bad"].tolist(), [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
