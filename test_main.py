import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pickle
from main import (
    load_embeddings, fuzzy_match_shows, cosine_similarity, create_random_show_name,
    request_image_generation, fetch_image_status, download_and_open_image, generate_image
)

# Mocked version of load_embeddings to simulate data
def mocked_load_embeddings(filepath):
    return {
        "Breaking Bad": np.array([0.1, 0.2, 0.3]),
        "Better Call Saul": np.array([0.1, 0.2, 0.4]),
        "Stranger Things": np.array([0.5, 0.6, 0.7])
    }

class TestShowAISuggest(unittest.TestCase):

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data=pickle.dumps(mocked_load_embeddings("/path/to/embeddings.pkl")))
    def test_load_embeddings(self, mock_open):
        embeddings = load_embeddings("/Users/anthonyghandour/Desktop/ShowAISuggest/src/data/embeddings.pkl")
        self.assertEqual(len(embeddings), 3)
        self.assertIn("Breaking Bad", embeddings)
        self.assertEqual(embeddings["Breaking Bad"].tolist(), [0.1, 0.2, 0.3])

    def test_fuzzy_match_shows(self):
        all_show_titles = ["Breaking Bad", "Better Call Saul", "Stranger Things"]
        user_input_shows = ["breaking bad", "stranger things"]
        matched_shows = fuzzy_match_shows(user_input_shows, all_show_titles)
        self.assertEqual(matched_shows, ["Breaking Bad", "Stranger Things"])

    def test_cosine_similarity(self):
        vec_a = np.array([1, 2, 3])
        vec_b = np.array([4, 5, 6])
        similarity = cosine_similarity(vec_a, vec_b)
        expected_similarity = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        self.assertAlmostEqual(similarity, expected_similarity)

    def test_create_random_show_name(self):
        source_shows = ["Breaking Bad", "Better Call Saul"]
        random_show_name = create_random_show_name(source_shows)
        self.assertTrue(isinstance(random_show_name, str))
        self.assertGreater(len(random_show_name), 0)

    @patch('requests.post')
    def test_request_image_generation(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"body": {"orderId": "12345"}}
        mock_post.return_value = mock_response

        prompt = "A scene from Breaking Bad"
        order_id = request_image_generation(prompt)
        self.assertEqual(order_id, "12345")

    @patch('requests.post')
    def test_fetch_image_status(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"body": {"status": "completed", "output": "http://example.com/image.jpg"}}
        mock_post.return_value = mock_response

        order_id = "12345"
        image_url = fetch_image_status(order_id)
        self.assertEqual(image_url, "http://example.com/image.jpg")

    @patch('requests.get')
    @patch('os.system')
    def test_download_and_open_image(self, mock_system, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'image data'
        mock_get.return_value = mock_response

        image_url = "http://example.com/image.jpg"
        filename = "test_image.jpg"
        download_and_open_image(image_url, filename)
        mock_get.assert_called_with(image_url)
        mock_system.assert_called_with(f'open {filename}')

    @patch('main.request_image_generation', return_value="12345")
    @patch('main.fetch_image_status', return_value="http://example.com/image.jpg")
    @patch('main.download_and_open_image')
    def test_generate_image(self, mock_download, mock_fetch, mock_request):
        prompt = "A scene from Breaking Bad"
        filename = generate_image(prompt)
        self.assertEqual(filename, "A_scene_from_Breaking_Bad.jpg")
        mock_request.assert_called_with(prompt.strip())
        mock_fetch.assert_called_with("12345")
        mock_download.assert_called_with("http://example.com/image.jpg", "A_scene_from_Breaking_Bad.jpg")

if __name__ == "__main__":
    unittest.main()
