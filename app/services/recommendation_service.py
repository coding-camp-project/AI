from google import genai
from google.genai import types
from app.core.config import settings

VALID_DISEASES = {'obesitas', 'diabetes', 'hipertensi', 'asam_urat', 'kolesterol'}

MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-pro-latest",
    "gemini-flash-latest"
]

async def get_food_recommendation(user_profile: dict = None, food_name: str = None, nutrition_data: dict = None) -> str:
    """
    Menghasilkan rekomendasi nutrisi atau makanan menggunakan Gemini API
    berdasarkan profil pengguna (umur, berat, tujuan, riwayat penyakit), 
    makanan tertentu, dan data nutrisinya.
    """
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    system_instruction = (
        "Anda adalah Nutrify AI, ahli gizi profesional. "
        "Berikan rekomendasi makanan, saran kalori, dan analisis nutrisi yang akurat "
        "berdasarkan data pengguna atau makanan yang diberikan."
    )
    
    # Membangun prompt berdasarkan data yang ada
    prompt = "Tolong berikan rekomendasi dan evaluasi nutrisi.\n"
    if user_profile:
        prompt += f"Profil/Kondisi Pengguna: {user_profile}\n"
    if food_name:
        prompt += f"Makanan yang sedang dipertimbangkan: {food_name}\n"
    if nutrition_data:
        prompt += f"Data Nutrisi Keseluruhan: {nutrition_data}\n"
        
    last_error = None
    
    for model_name in MODELS:
        try:
            print(f"Mendapatkan rekomendasi menggunakan model: {model_name}")
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            )
            if response.text:
                return response.text
        except Exception as e:
            print(f"Model {model_name} gagal: {str(e)}")
            last_error = e

    print("Semua model Gemini gagal digunakan untuk rekomendasi.")
    raise Exception(f"Gagal mendapatkan rekomendasi dari Gemini: {str(last_error)}")
