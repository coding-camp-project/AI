from google import genai
from google.genai import types
from app.core.config import settings

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash"
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
            last_error = e

    raise Exception(f"Gagal mendapatkan rekomendasi dari Gemini: {str(last_error)}")
