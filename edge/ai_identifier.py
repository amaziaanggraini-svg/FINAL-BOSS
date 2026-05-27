"""
ai_identifier.py — Smartbox Lost & Found
Mengidentifikasi nama dan deskripsi barang dari gambar menggunakan Gemini Vision API.
"""
import json       # <-- DETAIL 1: Dipindahkan ke paling atas agar bersifat global
import os
import base64
import logging
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv  # <-- DETAIL 2: Ditambahkan untuk membaca file .env

# Jalankan fungsi load_dotenv() untuk memuat isi file .env ke dalam sistem
load_dotenv()

log = logging.getLogger("Smartbox.AI")

# Ambil API key dari environment variable (.env di laptop Anda atau config vars di hosting cloud)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Konfigurasi client Gemini menggunakan key yang telah diambil
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3.1-flash-lite")  # Menggunakan model hemat kuota dan cepat

# Prompt panduan untuk AI Gemini
SYSTEM_PROMPT = """
Kamu adalah sistem identifikasi barang temuan untuk Lost & Found box.
Analisis gambar ini dan identifikasi barang yang ada di dalamnya.

Jawab HANYA dalam format JSON berikut (tanpa penjelasan tambahan, tanpa markdown):
{
  "nama": "<nama singkat barang, maks 5 kata>",
  "kategori": "<salah satu: elektronik / dokumen / aksesoris / pakaian / tas / kunci / lainnya>",
  "deskripsi": "<deskripsi singkat 1-2 kalimat, warna, kondisi, ciri khas>"
}

Jika tidak ada barang terlihat jelas, jawab:
{"nama": "tidak teridentifikasi", "kategori": "lainnya", "deskripsi": "Tidak ada barang yang dapat diidentifikasi dengan jelas."}
"""


def identify_item(image_path: Path) -> tuple[str, str]:
    """
    Identifikasi barang dari file gambar menggunakan Gemini Vision.

    Args:
        image_path: Path ke file gambar JPEG/PNG

    Returns:
        Tuple (nama_barang, deskripsi_barang)
    """
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY tidak diset! Jalankan: export GEMINI_API_KEY=your_key atau buat file .env")
        return "tidak teridentifikasi", "API key tidak tersedia."

    # DETAIL 3: Set variabel response menjadi None di awal agar aman dibaca jika terjadi eror koneksi
    response = None 

    try:
        log.info(f"Mengunggah gambar ke Gemini: {image_path}")

        with open(image_path, "rb") as f:
            image_data = f.read()

        # Mengirimkan data gambar (Base64) beserta instruksi Prompt ke Gemini
        response = model.generate_content([
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_data).decode("utf-8")
            },
            SYSTEM_PROMPT
        ])

        raw_text = response.text.strip()

        # Membersihkan kode jika Gemini tidak sengaja membungkus respons dengan markdown (```json ... ```)
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        # Mengubah teks mentah dari Gemini menjadi objek JSON Python
        result = json.loads(raw_text)
        nama      = result.get("nama", "tidak teridentifikasi")
        deskripsi = result.get("deskripsi", "")

        log.info(f"Identifikasi berhasil: {nama}")
        return nama, deskripsi

    except json.JSONDecodeError as e:
        # Jika respon gagal diubah menjadi JSON, ambil teks respon aman (jika ada)
        res_text = response.text[:200] if response else "Tidak ada respon dari server"
        log.error(f"Gagal parse JSON dari Gemini: {e}. Response: {res_text}")
        return "tidak teridentifikasi", "Gagal memproses respons AI."

    except Exception as e:
        log.error(f"Error Gemini API: {e}")
        return "tidak teridentifikasi", f"Error: {str(e)}"