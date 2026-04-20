import os
from flask import Flask, render_template, request, url_for
from deeplearning.deep_learning_service import predict

app = Flask(__name__)
# Ganti folder uploads ke dalam static agar bisa diakses langsung via URL
os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)

# Data Penyakit untuk Mapping Hasil Model
INFO_PENYAKIT = {
    "leaf spot": {
        "nama": "Bercak Daun (Leaf Spot)",
        "penjelasan": "Penyakit yang ditandai dengan munculnya bercak-bercak berwarna coklat atau hitam pada permukaan daun.",
        "penyebab": "Infeksi jamur Cercospora atau bakteri patogen yang berkembang biak pada lingkungan lembab.",
        "solusi": "Pangkas bagian daun yang terinfeksi. Semprotkan fungisida berbahan aktif tembaga dan pastikan jarak tanam tidak terlalu rapat untuk sirkulasi udara yang baik."
    },
    "leaf curl": {
        "nama": "Daun Keriting (Leaf Curl)",
        "penjelasan": "Daun mengalami perubahan bentuk, melengkung atau mengerut, biasanya disertai pertumbuhan tanaman yang terhambat.",
        "penyebab": "Infeksi virus (seperti Begomovirus) yang sering disebarkan oleh kutu kebul atau hama penghisap lainnya.",
        "solusi": "Kendalikan vektor serangga penyebab hama (kutu kebul). Cabut dan musnahkan tanaman yang terinfeksi parah agar tidak menular."
    },
    "whitefly": {
        "nama": "Kutu Kebul (Whitefly)",
        "penjelasan": "Adanya serangga kecil bersayap putih bergerombol di balik daun. Mereka menghisap cairan tanaman sehingga daun menguning.",
        "penyebab": "Serangga Bemisia tabaci yang berkembang cepat di kondisi hangat dan kering.",
        "solusi": "Gunakan perangkap kuning (yellow sticky trap). Semprotkan insektisida nabati (minyak nimba) atau gunakan predator alami."
    },
    "yellowish": {
        "nama": "Menguning (Yellowish / Chlorosis)",
        "penjelasan": "Daun tanaman memudar warnanya menjadi kuning secara keseluruhan atau pada bagian tertentu (klorosis).",
        "penyebab": "Bisa disebabkan oleh kekurangan nutrisi (seperti nitrogen), penyiraman berlebih, atau masalah perakaran.",
        "solusi": "Berikan pupuk yang tepat sesuai kekurangan unsur hara. Perbaiki sirkulasi drainase tanah agar akar tidak terendam air."
    },
    "healthy": {
        "nama": "Tanaman Sehat (Healthy)",
        "penjelasan": "Tanaman terlihat hijau, segar, dan tidak menunjukkan gejala penyakit atau hama.",
        "penyebab": "Perawatan dan kondisi lingkungan yang sudah sangat baik.",
        "solusi": "Pertahankan rutinitas penyiraman, pemupukan, dan pemantauan tanaman yang ada saat ini."
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diagnosa', methods=['POST'])
def diagnosa():
    if 'image' not in request.files:
        return "Tidak ada file yang diupload", 400
    
    file = request.files['image']
    if file.filename == '':
        return "Tidak ada file yang dipilih", 400
        
    if file:
        # Simpan file di dalam static/uploads
        filename = file.filename
        filepath = os.path.join('static', 'uploads', filename)
        file.save(filepath)
        
        # 1. Panggil fungsi predict dari model Deep Learning
        predicted_class, conf = predict(filepath)
        
        # Validasi jika model gagal dimuat
        if not predicted_class:
            return "Model Deep Learning belum dilatih. Latih model lewat deep_learning_service.py terlebih dahulu.", 500
            
        # 2. Sesuaikan tingkat kepercayaan (confidence)
        kepercayaan = f"{conf:.2f}%"
        
        # 3. Cari informasi penyakit dari dictionary berdasarkan class hasil prediksinya
        hasil = INFO_PENYAKIT.get(predicted_class.lower(), {
            "nama": predicted_class.title(),
            "penjelasan": f"Mendeteksi indikasi profil {predicted_class}.",
            "penyebab": "Tidak tersedia keterangan penyebab detail dari sistem.",
            "solusi": "Pantau terus tanaman anda untuk gejala berikutnya."
        })
        
        return render_template('result.html', 
                               image_path=url_for('static', filename='uploads/' + filename),
                               nama_penyakit=hasil['nama'],
                               penjelasan=hasil['penjelasan'],
                               penyebab=hasil['penyebab'],
                               solusi=hasil['solusi'],
                               confidence=kepercayaan)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)