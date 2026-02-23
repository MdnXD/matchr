from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)

# =========================
# KONFIGURASI DATABASE
# =========================
DB_CONFIG = {
    "host": "srv1785.hstgr.io",
    "user": "u324517725_matchr",
    "password": "Macacas27",
    "database": "u324517725_matchr",
    "charset": "utf8mb4",
    "connection_timeout": 10
}

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
    filtered = [w for w in tokens if len(w) >= 3 and not w.isdigit() and w not in STOP_WORDS]
    return ' '.join(filtered)

# =========================
# DATABASE
# =========================
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def get_all_judul():
    conn = get_db_connection()
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

def get_all_embeddings():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT e.id_judul, e.embedding, j.judul, j.sumber
        FROM embedding_judul e
        JOIN judul_skripsi j ON e.id_judul = j.id_judul
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def save_embedding(id_judul, embedding_vector):
    conn = get_db_connection()
    cursor = conn.cursor()
    embedding_json = json.dumps(embedding_vector.tolist())
    cursor.execute("""
        INSERT INTO embedding_judul (id_judul, embedding, dimensi)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE embedding = VALUES(embedding), dimensi = VALUES(dimensi)
    """, (id_judul, embedding_json, len(embedding_vector)))
    conn.commit()
    cursor.close()
    conn.close()

def train_vectorizer():
    all_judul = get_all_judul()
    if not all_judul:
        return None, []
    processed = [preprocess(row['judul']) for row in all_judul]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.7,
        sublinear_tf=True,
        norm='l2',
        use_idf=True,
        smooth_idf=True,
    )
    vectorizer.fit(processed)
    return vectorizer, all_judul

# =========================
# ROUTES
# =========================
@app.route('/health', methods=['GET'])
def health():
    try:
        conn = get_db_connection()
        conn.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "ok",
        "database": db_status
    })

@app.route('/similarity', methods=['POST'])
def similarity():
    try:
        data = request.get_json()
        judul_input = data.get('judul', '')
        top_n = data.get('top_n', 5)

        if not judul_input:
            return jsonify({"error": "Judul tidak boleh kosong"}), 400

        # Train vectorizer dari database
        vectorizer, all_judul = train_vectorizer()
        if not vectorizer:
            return jsonify({"error": "Tidak ada data judul di database"}), 404

        # Cek apakah embedding sudah ada, kalau belum generate
        rows = get_all_embeddings()
        if not rows:
            # Generate semua embedding
            for row in all_judul:
                vec = vectorizer.transform([preprocess(row['judul'])]).toarray()[0]
                save_embedding(row['id_judul'], vec)
            rows = get_all_embeddings()

        # Hitung similarity
        processed = preprocess(judul_input)
        input_vector = vectorizer.transform([processed]).toarray()[0]

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

if __name__ == '__main__':
    app.run(debug=False)
