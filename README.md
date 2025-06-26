# movierecommend
# 🎥 Movie Recommender (SVD + Streamlit)

## 🔍 Overview
This project is a **Movie Recommendation app** built with:
- **SVD (scikit-surprise)** trained on the **MovieLens-20M dataset** (~800MB ratings file)
- A **Streamlit web interface** allowing:
  - User ratings
  - Personalized recommendations
  - "More like this" search
- **TMDb API** integration to show posters

---

## 🧠 Key Features
✅ Personalized recommendations using SVD (~0.78 RMSE)  
✅ Search for similar movies (TF-IDF + cosine similarity)  
✅ Interactive UI with ratings and movie posters  
✅ Handles large datasets efficiently (~800MB)  

---

## 🧰 Tech Stack
**Languages:** Python  
**Frameworks/Libraries:** scikit-surprise, Streamlit, scikit-learn, Pandas, NumPy  
**Tools:** Git, Kaggle, TMDb API, Colab, VSCode

---

## 📂 Setup Instructions
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/movie-recommender-svd.git
   cd movie-recommender-svd
2. Install dependencies:
     ```bash
     pip install -r requirements.txt
3. Run the app:
   ```bash
   streamlit run streamlit_app.py
## 📂 Dataset
This app requires `ratings_df.csv` (~667 MB) and `svd_model.pkl` (~690MB),their drive links shall be updated asap.



