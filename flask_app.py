from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
import json
import os
import re
import numpy as np
import time
import signal
import sys
import pickle
import threading
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

# =========================
# FLASK SETUP
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# KONFIGURASI
# =========================
DB_CONFIG = {
    "host": "srv1785.hstgr.io",
    "user": "u324517725_matchr",
    "password": "Macacas27",
    "database": "u324517725_matchr",
    "charset": "utf8mb4"
}

MODEL_PATH = "model/tfidf_vectorizer.pkl"
CHECK_INTERVAL = 3
running = True

def signal_handler(sig, frame):
    global running
    print_flush("\n[SHUTDOWN] Menghentikan service...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =========================
# STOPWORDS
# =========================
STOP_WORDS = {
    'yang', 'dan', 'atau', 'dari', 'ke', 'di', 'pada', 'untuk', 'dengan',
    'dalam', 'oleh', 'sebagai', 'terhadap', 'secara', 'ini', 'itu', 'adalah',
    'akan', 'telah', 'dapat', 'ada', 'juga', 'lebih', 'saat', 'antara',
    'agar', 'bagi', 'bila', 'dimana', 'kembali', 'kepada', 'lain', 'maka',
    'mereka', 'saja', 'sama', 'sampai', 'sedang', 'setiap', 'suatu',
    'tentang', 'tersebut', 'yaitu', 'atas', 'bawah', 'depan', 'belakang',
    'skripsi', 'tugas', 'akhir', 'judul', 'penelitian', 'proposal',
    'laporan', 'karya', 'ilmiah', 'akademik',
    'sistem', 'aplikasi', 'web', 'android', 'mobile', 'website', 'site',
    'implementasi', 'perancangan', 'analisis', 'rancang', 'bangun',
    'berbasis', 'menggunakan', 'metode', 'studi', 'kasus', 'case',
    'desain', 'design', 'pembuatan', 'pembangunan', 'pengembangan',
    'development', 'interface', 'user', 'admin', 'administrator',
    'data', 'database', 'informasi', 'information', 'base',
    'manajemen', 'management', 'monitoring', 'control', 'pengelolaan',
    'qr', 'code', 'barcode', 'online', 'offline', 'digital', 'teknologi',
    'komputer', 'computer', 'software', 'hardware', 'program',
    'framework', 'library', 'plugin', 'module', 'component',
    'crud', 'api', 'rest', 'json', 'xml', 'sql', 'mysql', 'postgresql',
    'php', 'python', 'java', 'javascript', 'laravel', 'codeigniter',
    'bootstrap', 'jquery', 'ajax', 'html', 'css',
    'login', 'register', 'registrasi', 'authentication', 'authorization',
    'akun', 'account', 'profile', 'profil', 'dashboard', 'panel',
    'perpustakaan', 'universitas', 'university', 'makassar',
    'hasanuddin', 'unhas', 'kampus', 'campus', 'sekolah', 'school',
    'mahasiswa', 'student', 'siswa', 'peserta', 'baru', 'lama',
    'smk', 'sma', 'smp', 'sd', 'sman', 'smkn', 'negeri', 'swasta',
    'pt', 'cv', 'ud', 'toko', 'perusahaan', 'company', 'firma',
    'koperasi', 'yayasan', 'lembaga', 'instansi', 'dinas',
    'mengelola', 'menentukan', 'mengidentifikasi', 'meningkatkan',
    'membuat', 'membangun', 'merancang', 'mengembangkan', 'membentuk',
    'menyusun', 'menganalisis', 'mengevaluasi', 'mengukur', 'menghitung',
    'mencari', 'menemukan', 'menyimpan', 'mengambil', 'menampilkan',
    'pengguna', 'pemakai', 'pelanggan', 'customer', 'client', 'klien',
    'pegawai', 'karyawan', 'employee', 'staff', 'petugas', 'operator',
    'tingkat', 'level', 'grade', 'kualitas', 'quality', 'efisiensi',
    'efektivitas', 'produktivitas', 'kinerja', 'performance',
    'waktu', 'time', 'proses', 'process', 'tahap', 'fase', 'step',
    'baik', 'buruk', 'tinggi', 'rendah', 'besar', 'kecil',
    'cepat', 'lambat', 'mudah', 'sulit', 'simple', 'kompleks',
    'laporan', 'report', 'cetak', 'print', 'export', 'import',
    'upload', 'download', 'backup', 'restore', 'update', 'delete',
    'insert', 'edit', 'view', 'search', 'filter', 'sort',
    'hal', 'cara', 'jenis', 'tipe', 'type', 'model', 'bentuk', 'macam',
    'aspek', 'bagian', 'part', 'elemen', 'komponen', 'fitur', 'feature',
    'dipa',
}

print_flush(f"[CONFIG] Stopwords: {len(STOP_WORDS)} words loaded")

# =========================
# GLOBAL VECTORIZER
# =========================
vectorizer_global = None

# =========================
# PREPROCESSING
# =========================
def preprocess(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    filtered = [w for w in tokens
                if len(w) >= 3
                and not w.isdigit()
                and w not in STOP_WORDS]
    return ' '.join(filtered)

# =========================
# DATABASE FUNCTIONS
# =========================
def test_database_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        print_flush("[OK] Database connected")
        return True
    except Error as e:
        print_flush(f"[ERROR] Database connection failed: {e}")
        return False

def get_all_judul():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id_judul, judul, sumber
            FROM judul_skripsi
            WHERE judul IS NOT NULL AND TRIM(judul) != ''
            ORDER BY id_judul
        """)
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        return data
    except Error as e:
        print_flush(f"[ERROR] get_all_judul: {e}")
        return []

def get_pending_judul():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT j.id_judul, j.judul, j.sumber
            FROM judul_skripsi j
            LEFT JOIN embedding_judul e ON j.id_judul = e.id_judul
            WHERE e.id_embedding IS NULL
            AND j.judul IS NOT NULL AND TRIM(j.judul) != ''
            ORDER BY j.id_judul
        """)
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        return data
    except Error as e:
        print_flush(f"[ERROR] get_pending_judul: {e}")
        return []

def save_embedding(id_judul, embedding_vector):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        embedding_json = json.dumps(embedding_vector.tolist())
        cursor.execute("""
            INSERT INTO embedding_judul (id_judul, embedding, dimensi)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                embedding = VALUES(embedding),
                dimensi = VALUES(dimensi)
        """, (id_judul, embedding_json, len(embedding_vector)))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Error as e:
        print_flush(f"[ERROR] Save embedding {id_judul}: {e}")
        return False

# =========================
# TF-IDF MODEL
# =========================
def load_or_train_tfidf(all_judul):
    global vectorizer_global
    print_flush("[MODEL] Processing titles with TF-IDF...")

    processed_texts = [preprocess(row['judul']) for row in all_judul]

    if not processed_texts:
        print_flush("[ERROR] No valid texts after preprocessing!")
        return None

    print_flush(f"[MODEL] Total documents: {len(processed_texts)}")

    if os.path.exists(MODEL_PATH):
        try:
            print_flush("[LOAD] Loading existing TF-IDF vectorizer...")
            with open(MODEL_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            print_flush(f"[OK] Loaded vectorizer with {len(vectorizer.vocabulary_)} features")
            print_flush("[RETRAIN] Retraining with all data...")
            vectorizer.fit(processed_texts)
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(vectorizer, f)
            print_flush("[OK] Vectorizer updated\n")
            vectorizer_global = vectorizer
            return vectorizer
        except Exception as e:
            print_flush(f"[WARN] Failed to load: {e}, training new...")
            try:
                os.remove(MODEL_PATH)
            except:
                pass

    print_flush("[TRAIN] Training new TF-IDF vectorizer...")
    try:
        vectorizer = TfidfVectorizer(
            max_features=None,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.7,
            sublinear_tf=True,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
        )
        vectorizer.fit(processed_texts)
        os.makedirs("model", exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(vectorizer, f)
        print_flush(f"[OK] Vectorizer trained! Vocabulary: {len(vectorizer.vocabulary_)} features\n")
        vectorizer_global = vectorizer
        return vectorizer
    except Exception as e:
        print_flush(f"[ERROR] Training failed: {e}")
        return None

def judul_to_vector(text, vectorizer):
    processed = preprocess(text)
    if not processed:
        return np.zeros(len(vectorizer.vocabulary_))
    return vectorizer.transform([processed]).toarray()[0]

def process_pending_embeddings(vectorizer):
    pending = get_pending_judul()
    if not pending:
        print_flush("[INFO] No pending embeddings")
        return 0
    print_flush(f"[EMBEDDING] Processing {len(pending)} titles...")
    success_count = 0
    for row in pending:
        try:
            vector = judul_to_vector(row['judul'], vectorizer)
            if save_embedding(row['id_judul'], vector):
                success_count += 1
        except Exception as e:
            print_flush(f"[ERROR] ID {row['id_judul']}: {e}")
    print_flush(f"[DONE] Saved {success_count}/{len(pending)} embeddings\n")
    return success_count

# =========================
# FLASK ROUTES
# =========================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "vectorizer": vectorizer_global is not None,
        "features": len(vectorizer_global.vocabulary_) if vectorizer_global else 0
    })

@app.route('/similarity', methods=['POST'])
def similarity():
    try:
        if vectorizer_global is None:
            return jsonify({"error": "Vectorizer belum siap, tunggu beberapa detik"}), 503

        data = request.get_json()
        judul_input = data.get('judul', '')
        top_n = data.get('top_n', 5)

        if not judul_input:
            return jsonify({"error": "Judul tidak boleh kosong"}), 400

        # Ambil semua embedding dari database
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT e.id_judul, e.embedding, j.judul, j.sumber
            FROM embedding_judul e
            JOIN judul_skripsi j ON e.id_judul = j.id_judul
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return jsonify({"error": "Tidak ada embedding di database"}), 404

        # Buat vector input
        processed = preprocess(judul_input)
        input_vector = vectorizer_global.transform([processed]).toarray()[0]

        # Hitung similarity
        results = []
        for row in rows:
            emb = np.array(json.loads(row['embedding']))
            if len(emb) != len(input_vector):
                continue
            norm_input = np.linalg.norm(input_vector)
            norm_emb = np.linalg.norm(emb)
            score = 0.0
            if norm_input > 0 and norm_emb > 0:
                score = float(np.dot(input_vector, emb) / (norm_input * norm_emb))
            results.append({
                "id_judul": row['id_judul'],
                "judul": row['judul'],
                "sumber": row['sumber'],
                "similarity": round(score * 100, 2)
            })

        results.sort(key=lambda x: x['similarity'], reverse=True)

        return jsonify({
            "judul_input": judul_input,
            "processed": processed,
            "total_compared": len(results),
            "top_results": results[:top_n]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =========================
# EMBEDDING SERVICE (background thread)
# =========================
def run_embedding_service():
    global running, vectorizer_global

    print_flush("\n[SERVICE] Starting embedding service...")

    if not test_database_connection():
        return

    all_judul = get_all_judul()
    if not all_judul:
        print_flush("[ERROR] No titles in database!")
        return

    print_flush(f"[SERVICE] Found {len(all_judul)} titles")
    vectorizer = load_or_train_tfidf(all_judul)

    if not vectorizer:
        print_flush("[ERROR] Vectorizer failed!")
        return

    process_pending_embeddings(vectorizer)

    print_flush("[SERVICE] ✓ Embedding service ready! Monitoring for new titles...\n")

    check_count = 0
    while running:
        try:
            time.sleep(CHECK_INTERVAL)
            check_count += 1
            pending = get_pending_judul()
            if pending:
                print_flush(f"[{datetime.now().strftime('%H:%M:%S')}] New titles detected!")
                all_judul = get_all_judul()
                vectorizer = load_or_train_tfidf(all_judul)
                if vectorizer:
                    process_pending_embeddings(vectorizer)
            else:
                if check_count % 20 == 0:
                    print_flush(f"[{datetime.now().strftime('%H:%M:%S')}] Service running...")
        except Exception as e:
            print_flush(f"[ERROR] {e}")
            time.sleep(CHECK_INTERVAL)

# =========================
# MAIN
# =========================
if __name__ == '__main__':
    print_flush("\n╔══════════════════════════════════════════════════╗")
    print_flush("║   MATCHR - TF-IDF Service + Flask API v11.0     ║")
    print_flush("╚══════════════════════════════════════════════════╝\n")

    # Jalankan embedding service di background thread
    embed_thread = threading.Thread(target=run_embedding_service, daemon=True)
    embed_thread.start()

    # Tunggu vectorizer siap sebelum Flask mulai
    print_flush("[WAIT] Waiting for vectorizer to initialize...")
    timeout = 60
    waited = 0
    while vectorizer_global is None and waited < timeout:
        time.sleep(1)
        waited += 1

    if vectorizer_global:
        print_flush(f"[OK] Vectorizer ready after {waited}s")
    else:
        print_flush("[WARN] Vectorizer not ready yet, Flask will start anyway")

    # Jalankan Flask
    print_flush("[OK] Flask API running on http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
