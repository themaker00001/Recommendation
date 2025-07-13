# Recommendation
| Component     | Tech                                                                                  |
| ------------- | ------------------------------------------------------------------------------------- |
| Frontend      | Streamlit                                                                             |
| Backend       | Python                                                                                |
| Vector Search | FAISS (cosine similarity)                                                             |
| Embeddings    | `sentence-transformers` (MiniLM)                                                      |
| Dataset       | [TMDB 5000 Movies (Kaggle)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) |
| Posters       | TMDB API                                                                              |
| Config        | `.env` with `python-dotenv`                                                           |

Setup Instructions
1. Clone the Repo
bash
Copy
Edit
git clone https://github.com/your-username/movie-recommender
cd movie-recommender
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Download the Dataset
Place the Kaggle file tmdb_5000_movies.csv inside the data/ folder:

kotlin
Copy
Edit
movie_recommender/
└── data/
    └── tmdb_5000_movies.csv
Dataset link: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

4. Setup .env
Create a .env file and add your TMDB API key:

bash
Copy
Edit
TMDB_API_KEY=your_tmdb_api_key_here
Get your API key here: https://www.themoviedb.org/settings/api

5. Run the App
bash
Copy
Edit
streamlit run app.py
## 🔥 Demo Preview

![App Screenshot](screenshots/demo.png)
<p align="center">
  <img src="App Screenshot/screenshots/demo.png" width="80%">
</p>

Upcoming Features (Ideas)
🎞️ YouTube trailer integration

💬 Chatbot-based recommendations (LLM)

❤️ Like/Dislike memory with SQLite

🧬 Overview-only or cast-based embeddings

🎯 User profiles & genre affinity
