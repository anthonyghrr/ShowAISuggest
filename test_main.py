
import unittest
from unittest.mock import patch, mock_open
import pickle
from main import load_embeddings, match_shows

class TestShowMatcher(unittest.TestCase):

    @patch("builtins.open", side_effect=mock_open(read_data=pickle.dumps({"Show1": [0.1, 0.2, 0.3]})))
    def test_load_embeddings(self, mock_file):
        embeddings = load_embeddings("data/embeddings.pkl")
        mock_file.assert_called_once_with("data/embeddings.pkl", "rb")
        self.assertEqual(len(embeddings), 1)
        self.assertIn("Show1", embeddings)
        self.assertEqual(embeddings["Show1"], [0.1, 0.2, 0.3])

    @patch("thefuzz.process.extractOne", return_value=("Show1", 90))
    def test_match_shows_found(self, _):
        user_input = ["Show1"]
        embeddings = {"Show1": [0.1, 0.2, 0.3]}
        matched_shows = match_shows(user_input, embeddings)
        self.assertEqual(matched_shows, ["Show1"])

    @patch("thefuzz.process.extractOne", return_value=("Show1", 75))
    def test_match_shows_no_match(self, _):
        user_input = ["Show1"]
        embeddings = {"Show1": [0.1, 0.2, 0.3]}
        matched_shows = match_shows(user_input, embeddings)
        self.assertEqual(matched_shows, [])

    @patch("builtins.input", return_value="y")
    @patch("builtins.print")
    def test_user_confirmation_valid(self, mock_print, mock_input):
        user_input = ["Show1"]
        embeddings = {"Show1": [0.1, 0.2, 0.3]}
        matched_shows = match_shows(user_input, embeddings)
        self.assertTrue(mock_input.called)
        mock_print.assert_called_with(f"Do you mean Show1? (y/n)")

    @patch("builtins.input", return_value="n")
    @patch("builtins.print")
    def test_user_confirmation_invalid(self, mock_print, _):
        user_input = ["Show1"]
        embeddings = {"Show1": [0.1, 0.2, 0.3]}
        matched_shows = match_shows(user_input, embeddings)
        mock_print.assert_called_with("Please re-enter the shows with correct spelling.")

    @patch("builtins.input", return_value="y")
    @patch("thefuzz.process.extractOne", return_value=("Show1", 90))
    def test_end_to_end_flow(self, mock_input, _):
        user_input = ["Show1"]
        embeddings = {"Show1": [0.1, 0.2, 0.3]}
        matched_shows = match_shows(user_input, embeddings)
        self.assertEqual(matched_shows, ["Show1"])
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="y")
    @patch("builtins.print")
    def test_end_to_end_no_matches(self, mock_input, mock_print):
        user_input = ["UnknownShow"]
        embeddings = {"Show1": [0.1, 0.2, 0.3]}
        matched_shows = match_shows(user_input, embeddings)
        self.assertEqual(matched_shows, [])
        mock_print.assert_called_with("Sorry, no matches found. Please check your spelling and try again.")

if __name__ == "__main__":
    unittest.main()
