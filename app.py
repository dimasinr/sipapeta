import os
from flask import Flask, render_template, request, url_for
import random

app = Flask(__name__)
# Ganti folder uploads ke dalam static agar bisa diakses langsung via URL
os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)

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
        
        # MOCKUP: Simulasi hasil prediksi Penyakit Tanaman
        penyakit_list = [
            {
                "nama": "Bercak Daun (Leaf Spot)",
                "penjelasan": "Penyakit yang ditandai dengan munculnya bercak-bercak berwarna coklat atau hitam pada permukaan daun.",
                "penyebab": "Infeksi jamur Cercospora atau bakteri patogen yang berkembang biak pada lingkungan lembab.",
                "solusi": "Pangkas bagian daun yang terinfeksi. Semprotkan fungisida berbahan aktif tembaga dan pastikan jarak tanam tidak terlalu rapat untuk sirkulasi udara yang baik."
            },
            {
                "nama": "Karat Daun (Leaf Rust)",
                "penjelasan": "Munculnya bintik-bintik menyerupai serbuk karat berwarna oranye kemerahan di bagian bawah daun.",
                "penyebab": "Jamur patogen dari ordo Pucciniales yang menyebar melalui spora di udara.",
                "solusi": "Gunakan varietas tanaman yang tahan. Hilangkan sisa tanaman yang terinfeksi dan aplikasikan fungisida sistemik."
            }
        ]
        
        hasil = random.choice(penyakit_list)
        kepercayaan = f"{random.uniform(85.0, 99.9):.2f}%"
        
        return render_template('result.html', 
                               image_path=url_for('static', filename='uploads/' + filename),
                               nama_penyakit=hasil['nama'],
                               penjelasan=hasil['penjelasan'],
                               penyebab=hasil['penyebab'],
                               solusi=hasil['solusi'],
                               confidence=kepercayaan)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)