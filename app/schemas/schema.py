from pydantic import BaseModel, Field

"""
    Disini kita buat dict si data statis, kalo datanya dinamis kayak endpoint /predict itukan bergantung pada user input atau endpoint POST magnkanya kalo endpointya GET saya buatkan disini kalo dinamis harus strict nanti.
"""


# INI buat request
class ManualItem(BaseModel):
    """
        Ini buat bangun model kasaran ketika user pake manual item
    """
    food_name: str = Field(..., description="Nama makanan (wajib persis sesuai / search-food)")
    quantity: float = Field(default=1.0, gt=0, description="Jumlah units")
    unit: str = Field(default='porsi', description="Statuan: porsi, potong, iris, gelas, dll")
    


    
# INI buat response
class NutritionInfo(BaseModel):
    """Informasi nutrisi dalam satu makanan"""
    calories: float
    protein: float
    fat: float
    carbohydrates: float
    sugar: float
    sodium: float
    fiber: float

class FoodCandidate(BaseModel):
    food_name: str
    nutrition_per_100g: NutritionInfo | None

class SearchFoodResponse(BaseModel):
    """Response endpoint buat /search-food"""
    query: str
    count: int
    candidates: list[FoodCandidate]