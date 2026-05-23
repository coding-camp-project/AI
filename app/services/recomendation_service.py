"""
    service untuk rekomendasi konsumsi makanan pada orang yang terdiagnosa penyakit
"""

VALID_DISEASES = {'obesitas', 'diabetes', 'hipertensi', 'asam_urat', 'kolesterol'}

# Ini adalah logika bisnis rekomendasi pada nutrify ai
def _check_diabetes(nutrition: dict) -> list[str]:
    issues = []         
    if nutrition['sugar'] >= 15:
        issues.append(f"Kandungan gula {nutrition['sugar']:.1f}g tergolong sa tinggi")
    elif nutrition['sugar'] >= 8:
        issues.append(f"Kandungan gula {nutrition['sugar']:.1f}g tergolong cukup tinggi")
    if nutrition['carbohydrates'] >= 40:
        issues.append(f"Kandungan karbohidrat {nutrition['carbohydrates']:.1f}g  cukup tinggi")
    return issues

def _check_hipertensi(nutrition: dict) -> list[str]:
    issues = []
    sodium = nutrition['sodium']
    if sodium >= 600:
        issues.append(f"Kandungan Sodium {sodium:.0f}mg sangat tinggi (batas WHO 2000mg/hari)")
    elif sodium >= 400:
        issues.append(f"Kandungan sodium {sodium:.0f}mg cukup tinggi")
    return issues

def _check_obesitas(nutrition: dict) -> list[str]:
    issues = []
    calories = nutrition['calories']
    if calories >= 300:
        issues.append(f"Kandungan kalori {nutrition['calories']:.0f} kcal tergolong tinggi")
    if nutrition['fat'] >= 15:
        issues.append(f"Kandungan lemak {nutrition['fat']:.1f}g tergolong tinggi")
    if nutrition['sugar'] >= 15:
        issues.append(f"Kandungan gula {nutrition['sugar']:.1f}g tinggi")
    return issues

def _check_asam_urat(nutrition: dict, food_name: str) -> list[str]:
    issues = []
    name_lower = food_name.lower().replace("_", " ")
    high_purin = ['daging', 'rawon', 'sate', 'burger sapi', 'ikan', 'seafood', 'udang', 'kerang', 'hati', 'usus']
    
    for kw in high_purin:
        if kw in name_lower:
            issues.append(f"Makanan ini tergolong tinggi purin (mengandung {kw})")
            break
    if nutrition['protein'] >= 20:
        issues.append(f"Kandungan protein hewani {nutrition['protein']:.1f}g cukup tinggi")
    return issues

def _check_kolesterol(nutrition: dict, food_name: str) -> list[str]:
    issues = []
    name_lower = food_name.lower().replace("_", " ")
    if nutrition['fat'] >= 15:
        issues.append(f"Kandungan lemak {nutrition['fat']:.1f}g tergolong tinggi")
        fried = ["goreng", "nugget", "martabak", "burger"]
        for kw in fried:
            if kw in name_lower:
                issues.append(f"Termasuk makanan tinggi lemak jenuh / digoreng")
                break
    return issues
    
_DISEASE_CHECKERS = {
    'diabetes': (lambda n, f: _check_diabetes(n), 'diabetes'),
    'hipertensi': (lambda n, f: _check_hipertensi(n), 'hipertensi'),
    'obesitas': (lambda n, f: _check_obesitas(n), 'obesitas'),
    'asam_urat': (_check_asam_urat, 'asam urat'),
    'kolesterol': (_check_kolesterol, 'kolesterol tinggi')
}


# fungsi membuat teks rekomendasi
def generate_recomendation(food_name: str, nutrition: dict | None, disease: str | None = None) -> str:
    """
        penjelasan argument:
            1. food_name => nama makanan di dataset
            2. nutrition => ini dict nutrisi yang tadi
            3. disease => didapat dari personalisasi pengguna
    """
    
    if nutrition is None:
        return "Data nutrisi tidak tersedia untuk makanan ini"
    
    display_name = food_name.replace("_", " ").title()
    cal = nutrition['calories']
    protein = nutrition['protein']
    fat = nutrition['fat']
    
    
    """
        ini dibuat untuk rekomendasi mode personal: maksudanya ada penyakit
    """
    if disease and disease in _DISEASE_CHECKERS:
        checker, label = _DISEASE_CHECKERS[disease]
        issues = checker(nutrition, food_name)
        
        if not issues:
            return (
                f"{display_name} relatif aman untuk kondisi {label} Anda. "
                f"Total: {cal:.0f} kcal, protein {protein:.1f}g, lemak {fat:.1f}g"
                f"Tetap perhatikan porsi konsumsi total harian."
            )
        elif len(issues) == 1:
            return (
                f"Untuk penderita {label}, {display_name} sebaiknya dikonsumsi "
                f"dengan hati-hati karena {issues[0]}."
                f"Disarankan kurangi porsi atau cari alternatif."
                
            )
        else:
            return (
                f"Untuk penderita {label}, {display_name} sebaiknya dibatasi."
                f"atau hindari karena: {', '.join(issues)}. "
                f"Konsultasikan dengan ahli gizi atau dokter umum untuk pengaturan diet yang tepat."
            )
    
    """
        Ini khusus untuk kalo tidak ada penyakit pengguna atau mode umum
    """
    if cal < 100:
        category = "Rendah kalori dan cocok untuk diet"
    elif cal < 250:
        category = "Cukup seimbang untuk konsumsi harian"
    else:
        category = "tinggi kalori, sebaiknya dikonsumsi dengan porsi terkontrol"
    
    protein_text = (
        "Cukup baik untuk mendukung pembentukan otot."
        if protein > 15
        else "Masih dalam batas wajar."
    )
    fat_text = (
        "Tergolong tinggi, perhatikan frekuensi konsumsi."
        if fat > 15
        else "Masih dalam batas wajar."
    )
    
    return (
        f"{display_name} termasuk makanan {category} "
        f"dengan {cal:.0f} kcal. "
        f"Kandungan protein {protein:.1f}g {protein_text} "
        f"Lemak sebesar {fat:.1f}g {fat_text}"
    )
        
        
        
    
