from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import re

app = Flask(__name__)

# --- LOAD THE ENGINE ---
# This loads your dataframe, tfidf matrix, and the KNN model at once
try:
    with open('ott_recommendation_engine.pkl', 'rb') as f:
        data = pickle.load(f)
        df = data['movie_db']
        tfidf = data['vectorizer']
        tfidf_matrix = data['matrix']
        model_knn = data['model']
    print("✅ Model Engine Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    query = request.json.get('query', '').strip().lower()
    if not query:
        return jsonify([])
    
    # Efficiently find matches in the title column
    matches = df[df['title'].str.lower().str.contains(query, na=False)]['title'].unique().tolist()
    return jsonify(matches[:8]) # Return top 8 matches for the dropdown

@app.route('/recommend', methods=['POST'])
def recommend():
    req_data = request.json
    target_title = req_data.get('title')
    target_ott = req_data.get('ott')
    target_type = req_data.get('type') # 'Movie', 'TV Show', or 'Both'

    try:
        # 1. Locate the input movie
        idx = df[df['title'].str.lower() == target_title.lower()].index[0]
        
        # 2. Query KNN for a large pool of neighbors (to allow for filtering)
        distances, indices = model_knn.kneighbors(tfidf_matrix[idx], n_neighbors=150)
        indices = indices.flatten()
        distances = distances.flatten()
        
        results = []
        sim_weight = 0.7
        imdb_weight = 0.3

        for i in range(1, len(indices)): # Skip the first one (it's the movie itself)
            res_idx = indices[i]
            
            # --- FILTER: CONTENT TYPE ---
            item_type = str(df['type'].iloc[res_idx])
            if target_type != "Both" and item_type.lower() != target_type.lower():
                continue
                
            # --- FILTER: OTT PLATFORM ---
            # Check if target_ott exists in the 'available_on' list
            ott_raw = str(df['available_on'].iloc[res_idx])
            if target_ott != "All" and target_ott.lower() not in ott_raw.lower():
                continue

            # --- HYBRID RANKING CALCULATION ---
            sim_score = 1 - distances[i]
            imdb_val = df['imdb_score'].iloc[res_idx]
            # Normalize IMDb score (0-10) to a 0-1 scale
            normalized_imdb = (imdb_val / 10.0) if pd.notnull(imdb_val) else 0.5
            
            hybrid_score = (sim_score * sim_weight) + (normalized_imdb * imdb_weight)
            
            # Clean platform names for display
            ott_display = re.sub(r"[\[\]']", "", ott_raw)

            results.append({
                "Title": df['title'].iloc[res_idx],
                "Score": str(imdb_val) if imdb_val > 0 else "N/A",
                "Year": str(int(df['release_year'].iloc[res_idx])) if pd.notnull(df['release_year'].iloc[res_idx]) else "N/A",
                "Type": item_type,
                "Match": f"{round(sim_score * 100, 1)}%",
                "Rank": hybrid_score, # Used for sorting
                "OTT": ott_display,
                "Desc": df['description'].iloc[res_idx]
            })

        # Re-sort results by hybrid score (Similarity + IMDb Quality)
        final_list = sorted(results, key=lambda x: x['Rank'], reverse=True)[:6]

        return jsonify(final_list)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)