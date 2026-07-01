

import re

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

DATA = {
    "tentang_binus": [
        "apa itu binus", "binus universitas apa", "jelaskan tentang bina nusantara",
        "binus itu kampus seperti apa", "sejarah singkat binus", "binus berdiri tahun berapa",
        "apa kepanjangan binus", "binus terkenal di bidang apa", "profil universitas bina nusantara",
        "binus swasta atau negeri", "binus itu universitas atau institut",
        "ceritakan soal kampus binus", "bina nusantara itu apa sih", "binus didirikan kapan",
        "binus akreditasinya apa", "binus universitas swasta terbaik bukan",
        "kenapa harus kuliah di binus", "keunggulan binus apa", "binus fokus di jurusan apa",
        "visi misi binus apa", "binus kampus IT ya", "tahu binus university ga",
        "binus reputasinya gimana", "apakah binus terakreditasi", "binus berdiri sejak tahun berapa",
    ],
    "program_studi": [
        "jurusan apa saja di binus", "program studi yang ada di binus apa aja",
        "binus punya jurusan apa", "ada prodi teknik informatika ga di binus",
        "fakultas di binus apa saja", "mau kuliah jurusan komputer di binus bisa ga",
        "daftar jurusan binus", "binus ada jurusan dkv tidak", "pilihan program studi binus",
        "jurusan it di binus apa namanya", "binus ada jurusan manajemen ga",
        "prodi data science ada di binus", "jurusan bisnis di binus apa",
        "binus ada hukum tidak", "ada jurusan komunikasi di binus", "jurusan paling favorit di binus apa",
        "binus buka jurusan apa aja tahun ini", "ada jurusan desain di binus",
        "binus ada cyber security ga", "mau ambil informatika di binus",
        "jurusan akuntansi ada di binus", "binus ada jurusan hospitality ga",
        "list program studi bina nusantara", "binus punya prodi AI ga", "jurusan teknik di binus apa saja",
    ],
    "lokasi_kampus": [
        "kampus binus dimana", "alamat binus dimana", "lokasi kampus bina nusantara",
        "binus ada di kota mana saja", "kampus binus alam sutera dimana",
        "binus ada cabang di luar jakarta ga", "gedung binus letaknya dimana",
        "binus terdekat dari saya dimana", "kampus binus jakarta alamatnya apa",
        "binus ada di bandung tidak", "binus ada di malang ga", "kampus binus semarang dimana",
        "binus bekasi alamatnya", "binus punya berapa kampus", "lokasi binus kemanggisan",
        "binus anggrek dimana", "kampus binus syahdan letaknya", "binus ada dimana aja sih",
        "binus tangerang dimana", "kampus binus paling besar dimana",
        "binus ada di surabaya ga", "alamat lengkap kampus binus", "binus dekat stasiun apa",
        "kampus utama binus dimana", "binus ada di jakarta barat ya",
    ],
    "biaya_kuliah": [
        "biaya kuliah binus berapa", "uang kuliah di binus mahal ga",
        "spp binus berapa", "berapa ukt binus", "biaya masuk binus berapaan",
        "uang pangkal binus berapa", "estimasi biaya kuliah binus per semester",
        "biaya per sks binus", "total biaya kuliah di binus", "harga kuliah binus",
        "biaya kuliah informatika binus", "berapa duit kuliah di binus",
        "biaya semester binus berapa", "kuliah di binus habis berapa",
        "rincian biaya kuliah binus", "biaya pendidikan binus", "kisaran biaya binus setahun",
        "binus mahal ga sih", "biaya kuliah dkv binus", "berapa biaya daftar ulang binus",
        "biaya kuliah binus online berapa", "uang gedung binus berapa",
        "biaya kuliah binus 2026", "cicilan biaya kuliah binus ada ga", "biaya kuliah termurah di binus",
    ],
    "pendaftaran": [
        "cara daftar binus gimana", "bagaimana mendaftar mahasiswa baru binus",
        "syarat masuk binus apa saja", "pendaftaran binus dimana", "mau daftar binus caranya",
        "kapan pendaftaran binus dibuka", "jalur masuk binus apa saja",
        "daftar online binus lewat mana", "prosedur pmb binus", "registrasi mahasiswa baru binus",
        "syarat pendaftaran binus", "cara masuk binus", "binus ada tes masuk ga",
        "dokumen untuk daftar binus apa", "gimana cara jadi mahasiswa binus",
        "pendaftaran binus online atau offline", "jalur tanpa tes binus ada ga",
        "kapan gelombang pendaftaran binus", "link pendaftaran binus apa",
        "binus terima jalur rapor ga", "cara registrasi binus", "alur pendaftaran binus seperti apa",
        "daftar binus butuh apa aja", "kapan deadline daftar binus", "mau ikut seleksi binus gimana",
    ],
    "beasiswa": [
        "ada beasiswa di binus ga", "beasiswa binus apa saja", "binus kasih beasiswa tidak",
        "cara dapat beasiswa binus", "syarat beasiswa binus", "beasiswa prestasi binus",
        "widia scholarship itu apa", "binus ada potongan biaya untuk yang berprestasi ga",
        "jenis beasiswa di bina nusantara", "info beasiswa binus",
        "beasiswa full di binus ada ga", "binus ada beasiswa olahraga ga",
        "cara apply beasiswa binus", "beasiswa untuk anak tidak mampu di binus",
        "binus ada beasiswa akademik", "beasiswa binus untuk maba",
        "gimana dapat keringanan biaya binus", "binus ada diskon biaya kuliah ga",
        "beasiswa binus berapa persen", "syarat dapat widia scholarship",
        "beasiswa non akademik binus", "binus ada bantuan biaya kuliah ga",
        "kuota beasiswa binus berapa", "beasiswa binus untuk lulusan sma", "binus ada beasiswa prestasi non akademik",
    ],
    "kalender_akademik": [
        "kapan kuliah dimulai di binus", "jadwal perkuliahan binus",
        "kalender akademik binus", "kapan uts dan uas binus", "semester binus mulai kapan",
        "kapan libur semester binus", "jadwal akademik tahun ajaran binus",
        "binus pakai sistem semester atau term", "kapan masuk kuliah binus",
        "tanggal mulai kelas binus", "kapan ujian akhir binus", "jadwal libur binus",
        "kapan registrasi mata kuliah binus", "binus berapa semester setahun",
        "kapan awal perkuliahan binus", "jadwal uts binus kapan", "kapan semester baru binus",
        "kalender perkuliahan binus", "kapan periode ujian binus", "kapan orientasi mahasiswa baru binus",
        "binus mulai kuliah bulan apa", "kapan pengisian krs binus", "jadwal akademik semester ganjil binus",
        "kapan binus libur kuliah", "kapan binus selesai semester",
    ],
    "kontak": [
        "nomor telepon binus berapa", "kontak admisi binus", "email binus apa",
        "cara menghubungi binus", "customer service binus dimana",
        "kontak student service binus", "hubungi binus lewat apa",
        "nomor cs binus", "alamat email pendaftaran binus", "live chat binus dimana",
        "kontak binus untuk tanya", "telepon admisi binus", "wa binus berapa",
        "mau tanya ke binus lewat mana", "kontak resmi binus", "email admissions binus",
        "nomor hotline binus", "binus bisa dihubungi dimana", "cara chat dengan binus",
        "kontak layanan mahasiswa binus", "nomor whatsapp binus", "social media binus apa",
        "instagram binus apa", "cara komplain ke binus", "menghubungi pihak binus gimana",
    ],
    "online_learning": [
        "binus online learning itu apa", "kuliah online di binus bisa ga",
        "ada kelas daring binus", "binus untuk karyawan yang kerja",
        "kuliah jarak jauh binus", "binus online untuk yang sudah bekerja",
        "kelas malam binus ada ga", "program online binus seperti apa",
        "bisa kuliah sambil kerja di binus", "binus online learning beda dengan reguler ga",
        "binus ada kuliah online", "kuliah daring binus gimana", "binus online itu gimana",
        "mau kuliah online di binus", "binus bol itu apa", "kelas online binus jadwalnya gimana",
        "binus online ijazahnya sama ga", "kuliah online binus untuk pekerja",
        "binus ada program jarak jauh", "kuliah fleksibel binus ada ga",
        "binus online learning biayanya gimana", "kuliah online binus diakui ga",
        "binus online kelasnya kapan", "bisa kuliah remote di binus", "binus online learning cocok buat karyawan ga",
    ],
    "fasilitas": [
        "fasilitas di binus apa saja", "binus ada perpustakaan ga",
        "lab komputer binus", "ada wifi di kampus binus", "fasilitas kampus bina nusantara",
        "binus ada asrama tidak", "ruang belajar di binus seperti apa",
        "fasilitas penunjang kuliah binus", "binus punya studio desain ga",
        "sarana dan prasarana binus", "binus ada kantin ga", "fasilitas olahraga di binus",
        "binus ada lab khusus ga", "ruang kelas binus seperti apa", "binus ada tempat parkir ga",
        "fasilitas mahasiswa binus apa aja", "binus ada coworking space ga",
        "perpustakaan binus bukanya kapan", "binus ada lab AI ga", "fasilitas kampus binus lengkap ga",
        "binus ada masjid atau musholla ga", "binus ada gym ga", "ada laboratorium di binus",
        "fasilitas teknologi di binus", "binus ada ruang diskusi ga",
    ],
}

ANSWERS = {
    "tentang_binus": "Bina Nusantara (Binus) adalah universitas swasta di Indonesia yang berdiri "
        "sejak 1981, dikenal kuat di bidang teknologi informasi, bisnis, desain, dan komunikasi.",
    "program_studi": "Binus punya banyak program studi: Teknik Informatika, Sistem Informasi, "
        "Data Science, Cyber Security, Manajemen, Akuntansi, DKV, Komunikasi, Hukum, Hospitality, dll. "
        "Daftar lengkap & terbaru ada di binus.ac.id.",
    "lokasi_kampus": "Binus punya beberapa kampus: Jakarta (Kemanggisan - Anggrek, Syahdan, Kijang), "
        "Alam Sutera (Tangerang), Bekasi, Bandung, Malang, dan Semarang. Detail alamat ada di binus.ac.id.",
    "biaya_kuliah": "Biaya kuliah di Binus berbeda per program studi dan jalur masuk, dan berubah tiap "
        "tahun ajaran. Untuk angka pasti, cek halaman biaya resmi atau hubungi Admissions Binus.",
    "pendaftaran": "Pendaftaran mahasiswa baru Binus dilakukan online lewat student.binus.ac.id atau "
        "datang ke Admissions. Umumnya butuh ijazah/rapor, identitas diri, dan mengikuti seleksi sesuai jalur.",
    "beasiswa": "Binus menyediakan beberapa beasiswa, mis. beasiswa prestasi akademik & non-akademik "
        "(Widia Scholarship, dll). Syarat dan kuotanya berubah tiap periode; pantau pengumuman resmi Binus.",
    "kalender_akademik": "Tahun akademik Binus terbagi dalam beberapa semester/term. Tanggal pasti "
        "perkuliahan, UTS, dan UAS ada di kalender akademik resmi yang dibagikan ke mahasiswa.",
    "kontak": "Pertanyaan resmi bisa diarahkan ke Binus Admissions melalui binus.ac.id (live chat, "
        "telepon, email). Mahasiswa aktif bisa hubungi Student Service Center di kampus masing-masing.",
    "online_learning": "Binus punya Binus Online Learning untuk kuliah daring yang fleksibel bagi "
        "pekerja. Program dan jadwalnya berbeda dari kelas reguler.",
    "fasilitas": "Fasilitas Binus antara lain perpustakaan, laboratorium komputer, studio desain, "
        "area kolaborasi, dan koneksi internet kampus. Ketersediaan asrama bervariasi per kampus.",
}



STOPWORDS = {
    "yang", "di", "ke", "dari", "dan", "atau", "untuk", "pada", "dengan", "ini", "itu",
    "apa", "apakah", "adalah", "ada", "saya", "kamu", "aku", "kita", "mau", "ingin",
    "tolong", "bisa", "gimana", "bagaimana", "berapa", "dimana", "kapan", "siapa",
    "kenapa", "mengapa", "nya", "kah", "dong", "ya", "sih", "deh", "kok", "saja",
    "aja", "buat", "juga", "lagi", "kalau", "kalo", "tentang", "seputar",
}

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _stemmer = StemmerFactory().create_stemmer()
    def stem(w): return _stemmer.stem(w)
    STEMMER_NAME = "Sastrawi"
except Exception:
    _PRE = ("meng", "meny", "men", "mem", "me", "peng", "peny", "pen", "pem", "pe",
            "ber", "ter", "di", "ke", "se")
    _SUF = ("kan", "an", "i", "nya", "lah", "kah")
    def stem(w):
        for s in _SUF:
            if w.endswith(s) and len(w) - len(s) >= 4:
                w = w[:-len(s)]; break
        for p in _PRE:
            if w.startswith(p) and len(w) - len(p) >= 4:
                w = w[len(p):]; break
        return w
    STEMMER_NAME = "fallback (Sastrawi tidak terpasang)"


def preprocess(text):
    text = text.lower()
    toks = re.findall(r"[a-z0-9]+", text)
    toks = [t for t in toks if t not in STOPWORDS]
    toks = [stem(t) for t in toks]
    return " ".join(t for t in toks if t)


def build_dataset():
    """Return (X_raw_preprocessed: list[str], y: list[str])."""
    X_raw, y = [], []
    for intent, samples in DATA.items():
        for s in samples:
            X_raw.append(preprocess(s))
            y.append(intent)
    return X_raw, y



class XGBStringWrapper(BaseEstimator, ClassifierMixin):
    """Pembungkus XGBoost agar bisa menerima label berupa teks.

    XGBoost hanya menerima label numerik 0..n. Wrapper ini meng-encode label
    teks -> angka saat fit, dan men-decode angka -> teks saat predict, sehingga
    bagian program lain (chatbot, ANSWERS) tetap memakai label teks seperti biasa.
    """
    def __init__(self, n_estimators=300, max_depth=4, learning_rate=0.3, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        from xgboost import XGBClassifier  
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self._model = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, eval_metric="mlogloss",
            random_state=self.random_state)
        self._model.fit(X, y_enc)
        self.classes_ = self._le.classes_     
        return self

    def predict(self, X):
        return self._le.inverse_transform(self._model.predict(X))

    def predict_proba(self, X):
        return self._model.predict_proba(X)
