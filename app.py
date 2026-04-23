import os
import time
import pickle
import re
import string
import numpy as np
import pandas as pd
import faiss
import nltk
import torch
import io
import csv
from flask import Flask, render_template, request, Response, url_for
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -----------------------------------------------------------------------------
# 1. SETUP APLIKASI DAN PEMUATAN MODEL (DILAKUKAN SEKALI SAAT STARTUP)
# -----------------------------------------------------------------------------

app = Flask(__name__)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ARTEFAK_DIR = 'artefak'

def load_artefak(path):
    print(f"Memuat artefak dari: {path}...")
    try:
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.index'):
            return faiss.read_index(path)
        else:
            return None
    except FileNotFoundError:
        print(f"ERROR: File tidak ditemukan di {path}")
        raise
    except Exception as e:
        print(f"ERROR saat memuat {path}: {e}")
        raise

print("--- Memuat Semua Model dan Artefak ---")
SBERT_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)

faiss_index = load_artefak(os.path.join(ARTEFAK_DIR, 'goodreads_faiss_ivfflat.index'))
df_display_metadata = load_artefak(os.path.join(ARTEFAK_DIR, 'df_metadata_for_app.csv'))
tfidf_vectorizer = load_artefak(os.path.join(ARTEFAK_DIR, 'tfidf_vectorizer.pkl'))
corpus_tfidf_matrix = load_artefak(os.path.join(ARTEFAK_DIR, 'tfidf_matrix.pkl'))
print("--- Semua Artefak Berhasil Dimuat ---")

# -----------------------------------------------------------------------------
# 2. FUNGSI HELPER (PREPROCESSING & PENCARIAN)
# -----------------------------------------------------------------------------

def preprocess_text_for_tfidf(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

def clean_text_sbert(text_input):
    if not isinstance(text_input, str): return ""
    text = text_input.lower()
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def parse_boolean_query(query):
    if ' AND ' in query:
        parts = query.split(' AND ', 1)
        return parts[0].strip(), 'AND', parts[1].strip()
    elif ' OR ' in query:
        parts = query.split(' OR ', 1)
        return parts[0].strip(), 'OR', parts[1].strip()
    else:
        return query, None, None

def _search_sbert_single(query, k=1000):
    cleaned_query = clean_text_sbert(query)
    query_embedding = sbert_model.encode([cleaned_query], convert_to_numpy=True)
    query_embedding_normalized = normalize(query_embedding, norm='l2', axis=1).astype(np.float32)
    distances, indices = faiss_index.search(query_embedding_normalized, k)
    valid_indices = indices.flatten()[indices.flatten() != -1]
    if len(valid_indices) == 0:
        return pd.DataFrame()
    results_df = df_display_metadata.iloc[valid_indices].copy()
    results_df['score'] = distances.flatten()[indices.flatten() != -1]
    return results_df

def search_sbert_boolean_func(full_query, page=1, per_page=10):
    sub1, op, sub2 = parse_boolean_query(full_query)
    
    if op is None:
        results_df = _search_sbert_single(sub1, k=100)
    else:
        results1_df = _search_sbert_single(sub1)
        results2_df = _search_sbert_single(sub2)
        
        merged_df = pd.DataFrame()
        if op == 'AND':
            if results1_df.empty or results2_df.empty: return {'results': [], 'total_results': 0, 'total_pages': 0, 'current_page': 1}
            combined_results = pd.concat([results1_df, results2_df])
            if combined_results.empty: return {'results': [], 'total_results': 0, 'total_pages': 0, 'current_page': 1}
            grouped = combined_results.groupby('book_id').agg(score=('score', 'sum'), count=('score', 'size'), **{col: (col, 'first') for col in df_display_metadata.columns if col != 'book_id'}).reset_index()
            bonus_multiplier = 1.5
            grouped['score'] = grouped.apply(lambda row: row['score'] * bonus_multiplier if row['count'] == 2 else row['score'], axis=1)
            merged_df = grouped.drop(columns=['count'])
        elif op == 'OR':
            combined_results = pd.concat([results1_df, results2_df])
            if combined_results.empty: return {'results': [], 'total_results': 0, 'total_pages': 0, 'current_page': 1}
            merged_df = combined_results.sort_values('score', ascending=False).drop_duplicates('book_id').reset_index(drop=True)
        results_df = merged_df

    if results_df.empty:
        return {'results': [], 'total_results': 0, 'total_pages': 0, 'current_page': 1}

    sorted_results = results_df.sort_values(by='score', ascending=False)
    
    total_results = len(sorted_results)
    total_pages = (total_results + per_page - 1) // per_page
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    
    paginated_df = sorted_results.iloc[start_index:end_index]

    return {
        'results': paginated_df.to_dict('records'),
        'total_results': total_results,
        'total_pages': total_pages,
        'current_page': page
    }

def search_tfidf_func(query, page=1, per_page=10):
    processed_query = preprocess_text_for_tfidf(query)
    query_vector = tfidf_vectorizer.transform([processed_query])
    cosine_similarities = cosine_similarity(query_vector, corpus_tfidf_matrix).flatten()
    
    relevant_indices = np.where(cosine_similarities > 0.01)[0]
    if len(relevant_indices) == 0:
        return {'results': [], 'total_results': 0, 'total_pages': 0, 'current_page': 1}
        
    sorted_indices = relevant_indices[np.argsort(cosine_similarities[relevant_indices])[::-1]]
    
    total_results = len(sorted_indices)
    total_pages = (total_results + per_page - 1) // per_page
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    
    paginated_indices = sorted_indices[start_index:end_index]
    
    results = df_display_metadata.iloc[paginated_indices].copy()
    results['score'] = cosine_similarities[paginated_indices]

    return {
        'results': results.to_dict('records'),
        'total_results': total_results,
        'total_pages': total_pages,
        'current_page': page
    }

def search_exact_func(full_query, page=1, per_page=10):
    sub1, op, sub2 = parse_boolean_query(full_query)
    search_column = df_display_metadata['title'].fillna('') + ' ' + df_display_metadata['author'].fillna('')
    
    empty_result = {'results': [], 'total_results': 0, 'total_pages': 0, 'current_page': 1}
    
    if op is None:
        if not sub1: return empty_result
        mask = search_column.str.contains(sub1, case=False, na=False, regex=False)
    elif op == 'AND':
        if not sub1 or not sub2: return empty_result
        mask1 = search_column.str.contains(sub1, case=False, na=False, regex=False)
        mask2 = search_column.str.contains(sub2, case=False, na=False, regex=False)
        mask = mask1 & mask2
    elif op == 'OR':
        if not sub1 and not sub2: return empty_result
        mask1 = search_column.str.contains(sub1, case=False, na=False, regex=False)
        mask2 = search_column.str.contains(sub2, case=False, na=False, regex=False)
        mask = mask1 | mask2
    else:
        return empty_result

    all_results_df = df_display_metadata[mask].copy()
    all_results_df['score'] = 1.0
    
    total_results = len(all_results_df)
    total_pages = (total_results + per_page - 1) // per_page
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    
    paginated_df = all_results_df.iloc[start_index:end_index]

    return {
        'results': paginated_df.to_dict('records'),
        'total_results': total_results,
        'total_pages': total_pages,
        'current_page': page
    }

# -----------------------------------------------------------------------------
# 3. ROUTING APLIKASI (HALAMAN WEB)
# -----------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('query', '')
        model_type = request.form.get('model', 'sbert')
        page = 1
    else: 
        query = request.args.get('query', '')
        model_type = request.args.get('model', 'sbert')
        page = request.args.get('page', 1, type=int)

    search_data = {}
    search_time = 0
    start_time = time.time()
    
    if model_type == 'sbert':
        search_data = search_sbert_boolean_func(query, page=page)
    elif model_type == 'tfidf':
        search_data = search_tfidf_func(query, page=page)
    elif model_type == 'exact':
        search_data = search_exact_func(query, page=page)
        
    end_time = time.time()
    search_time = (end_time - start_time) * 1000

    return render_template(
        'results.html', 
        query=query, 
        model_type_internal=model_type, # Untuk link paginasi
        model_type_display=model_type.upper().replace('_', ' '), # Untuk tampilan
        search_time=f"{search_time:.2f}",
        results=search_data.get('results', []),
        total_results=search_data.get('total_results', 0),
        total_pages=search_data.get('total_pages', 0),
        current_page=search_data.get('current_page', 1)
    )

@app.route('/export')
def export():
    query = request.args.get('query', '')
    model_type = request.args.get('model', 'sbert')
    export_format = request.args.get('format', 'txt')

    # Mengambil 10 hasil teratas untuk ekspor
    per_page_export = 10
    
    search_data = {}
    if model_type == 'sbert':
        search_data = search_sbert_boolean_func(query, page=1, per_page=per_page_export)
    elif model_type == 'tfidf':
        search_data = search_tfidf_func(query, page=1, per_page=per_page_export)
    elif model_type == 'exact':
        search_data = search_exact_func(query, page=1, per_page=per_page_export)
    
    results_list = search_data.get('results', [])
    
    if not results_list:
        return "Tidak ada data untuk diekspor.", 404

    output = io.StringIO()
    safe_query_name = re.sub(r'[^a-zA-Z0-9]', '_', query[:30])
    filename = f"hasil_{model_type}_{safe_query_name}.{export_format}"
    
    if export_format == 'csv':
        df_results = pd.DataFrame(results_list)
        # Menambahkan 'desc' ke kolom ekspor CSV
        columns_to_export = ['title', 'author', 'desc', 'score', 'link']
        df_to_export = df_results[[col for col in columns_to_export if col in df_results.columns]]
        df_to_export.to_csv(output, index=False, quoting=csv.QUOTE_ALL)
        mimetype = 'text/csv'

    # --- PERUBAHAN DI SINI: Format TXT disesuaikan untuk menyertakan deskripsi ---
    elif export_format == 'txt':
        output.write(f"--- Hasil Pencarian untuk Query: '{query}' ---\n")
        output.write(f"--- Menggunakan Model: {model_type.upper()} ---\n\n")
        
        for i, book in enumerate(results_list):
            title = book.get('title', 'N/A')
            author = book.get('author', 'N/A')
            
            # Menangani deskripsi yang mungkin kosong atau bukan string
            desc = book.get('desc', 'Deskripsi tidak tersedia.')
            if not isinstance(desc, str) or not desc.strip():
                desc = 'Deskripsi tidak tersedia.'
            
            output.write(f"## Hasil #{i+1}\n")
            output.write(f"Judul: {title}\n")
            output.write(f"Penulis: {author}\n")
            output.write(f"Deskripsi: {desc}\n\n") # Menambahkan baris deskripsi
            
        mimetype = 'text/plain'
        
    else:
        return "Format tidak didukung.", 400

    return Response(
        output.getvalue(),
        mimetype=mimetype,
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )

if __name__ == '__main__':
    app.run(debug=True)