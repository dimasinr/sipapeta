import os
from flask import Flask, render_template, request, url_for
from deeplearning.deep_learning_service import predict

app = Flask(__name__)

# Data Penyakit untuk Mapping Hasil Model
INFO_PENYAKIT = {
    "leaf spot": {
        "nama": "Bercak Daun (Leaf Spot)",
        "penjelasan": "Penyakit yang ditandai dengan munculnya bercak-bercak berwarna coklat atau hitam pada permukaan daun.",
        "penyebab": "Biasanya karena infeksi jamur Cercospora atau bakteri yang suka banget berkembang biak di tempat yang lembab.",
        "solusi": "Potong aja daun yang udah kena biar nggak nyebar. Terus, semprot pakai fungisida berbahan aktif tembaga. Jangan lupa, kasih jarak tanam yang agak longgar biar sirkulasi udaranya lancar."
    },
    "leaf curl": {
        "nama": "Daun Keriting (Leaf Curl)",
        "penjelasan": "Daun mengalami perubahan bentuk, melengkung atau mengerut, biasanya disertai pertumbuhan tanaman yang terhambat.",
        "penyebab": "Ini gara-gara infeksi virus (kayak Begomovirus) yang biasanya dibawa dan disebarin sama kutu kebul atau hama penghisap lainnya.",
        "solusi": "Basmi dulu hama pembawa virusnya, kayak si kutu kebul. Kalau ada tanaman yang udah parah banget keritingnya, mending langsung cabut dan buang aja biar nggak nular ke yang lain."
    },
    "whitefly": {
        "nama": "Kutu Kebul (Whitefly)",
        "penjelasan": "Adanya serangga kecil bersayap putih bergerombol di balik daun. Mereka menghisap cairan tanaman sehingga daun menguning.",
        "penyebab": "Gara-gara serangga kutu kebul (Bemisia tabaci) yang cepet banget berkembang biak pas cuaca lagi hangat dan kering.",
        "solusi": "Pasang perangkap kuning (yellow sticky trap) di sekitar tanaman. Bisa juga semprotin insektisida nabati kayak minyak nimba, atau manfaatin serangga predator alami buat mangsa mereka."
    },
    "yellowish": {
        "nama": "Menguning (Yellowish / Chlorosis)",
        "penjelasan": "Daun tanaman memudar warnanya menjadi kuning secara keseluruhan atau pada bagian tertentu (klorosis).",
        "penyebab": "Biasanya sih karena tanaman kurang nutrisi (terutama nitrogen), terlalu banyak disiram air, atau ada masalah di bagian akarnya.",
        "solusi": "Kasih pupuk yang pas buat nambahin nutrisi yang kurang. Terus, pastikan drainase tanahnya bagus supaya airnya nggak menggenang dan bikin akar membusuk."
    },
    "healthy": {
        "nama": "Tanaman Sehat (Healthy)",
        "penjelasan": "Tanaman terlihat hijau, segar, dan tidak menunjukkan gejala penyakit atau hama.",
        "penyebab": "Perawatan kamu udah top banget, dan kondisi lingkungannya juga sangat mendukung.",
        "solusi": "Lanjutin aja rutinitas nyiram, mupuk, sama ngecek tanaman kayak yang udah kamu lakuin sekarang. Udah bagus kok."
    },
    "anthracnose": {
        "nama": "Antraknosa (Anthracnose / Patek)",
        "penjelasan": "Penyakit yang ditandai dengan munculnya bercak melingkar basah kehitaman pada buah cabai, lambat laun mengering, mengerut, dan menyebabkan buah membusuk serta gugur.",
        "penyebab": "Kena infeksi jamur Colletotrichum capsici. Jamur ini cepet banget nyebarnya kalau cuaca lagi lembab dan hangat.",
        "solusi": "Kalau ada buah cabai yang kena, langsung petik dan buang jauh-jauh. Usahain jangan nyiram air langsung ke daun atau buah (mending pakai irigasi tetes), terus semprot fungisida yang ada bahan aktif mankozeb atau tembaga hidroksida."
    },
    "veinal mottle virus": {
        "nama": "Virus Mottle Pembuluh Daun (Veinal Mottle Virus)",
        "penjelasan": "Gejala berupa warna belang hijau tua dan hijau muda (mosaik) di sepanjang pembuluh vena daun, terkadang disertai daun yang sedikit mengerut atau menyempit.",
        "penyebab": "Terkena infeksi Chilli Veinal Mottle Virus (ChiVMV) yang biasanya dibawa dan ditularin sama kutu daun (Aphis gossypii).",
        "solusi": "Atasi dulu kutu daunnya pakai insektisida nabati atau kimia. Terus, kalau ada tanaman yang udah nunjukin gejala virus ini, mending langsung cabut dan bakar aja biar nggak nularin tanaman cabai yang masih sehat."
    },
    "non-chilli": {
        "nama": "Bukan Tanaman Cabai (Non-Chilli)",
        "penjelasan": "Gambar yang diupload bukan merupakan tanaman cabai.",
        "penyebab": "Gambar yang kamu masukin sepertinya bukan tanaman cabai deh.",
        "solusi": "Upload Foto Tanaman Cabai."
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
        
        predicted_class, conf, all_predictions = predict(filepath)
        
        # Validasi jika model gagal dimuat
        if not predicted_class:
            return "Model Deep Learning belum dilatih. Latih model lewat deep_learning_service.py terlebih dahulu.", 500
            
        # 2. Sesuaikan tingkat kepercayaan (confidence)
        kepercayaan = f"{conf:.2f}%"
        
        # 3. Cari informasi penyakit dari dictionary berdasarkan class hasil prediksinya
        # hasil = INFO_PENYAKIT.get(predicted_class.lower(), {
        #     "nama": predicted_class.title(),
        #     "penjelasan": f"Mendeteksi indikasi profil {predicted_class}.",
        #     "penyebab": "Tidak tersedia keterangan penyebab detail dari sistem.",
        #     "solusi": "Pantau terus tanaman anda untuk gejala berikutnya."
        # })
        hasil = INFO_PENYAKIT.get(predicted_class.lower(), {})  # Using empty dict as default
        print(hasil)

        return render_template('result.html', 
                               image_path=url_for('static', filename='uploads/' + filename),
                               nama_penyakit=hasil.get('nama', predicted_class.title()),
                               penjelasan=hasil.get('penjelasan', f"Mendeteksi indikasi profil {predicted_class}."),
                               penyebab=hasil.get('penyebab', "Tidak tersedia keterangan penyebab detail dari sistem."),
                               solusi=hasil.get('solusi', "Pantau terus tanaman anda untuk gejala berikutnya."),
                               confidence=kepercayaan,
                               predictions=all_predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)