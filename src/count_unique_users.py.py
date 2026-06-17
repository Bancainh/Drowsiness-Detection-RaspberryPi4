import pandas as pd
import re

FILE_PATH = r'D:\Drowsiness_Project\03_Final_Dataset ( File gộp dữ liệu cuối cùng )\ULTIMATE_MASTER_DATASET.csv'

print("🔍 Đang khởi động hệ thống lột xác siêu dữ liệu...")
df = pd.read_csv(FILE_PATH)

# Lấy cột chứa tên (ưu tiên User_Name, nếu không thì lấy Original_File_x)
col_name = 'User_Name' if 'User_Name' in df.columns else 'Original_File_x'

raw_names = df[col_name].dropna().astype(str).unique()
grouped_names = {}

for name in raw_names:
    # 1. Ép về chữ thường và xóa đuôi file
    name_clean = name.replace('.csv', '').strip().lower()
    
    # 2. Chém sạch các từ khóa nhiễu của môi trường (Cậu có thể thêm vào danh sách này)
    for noise in ['thieu_sang_', 'night_light_', '_deo_kinh', 'day_light_', 'lucsang']:
        name_clean = name_clean.replace(noise, '')
        
    # 3. Lấy phần chữ cái đầu tiên làm GỐC (bỏ qua số 1, 2, 3... phía sau)
    match = re.match(r'([a-z]+)', name_clean)
    base = match.group(1) if match else name_clean
    
    if base not in grouped_names:
        grouped_names[base] = []
    grouped_names[base].append(name)

print("\n" + "="*70)
print(f"📊 MÁY TÍNH DỰ ĐOÁN CÓ KHOẢNG: {len(grouped_names)} NGƯỜI (Cần tự đếm lại)")
print("="*70)

nghi_van = 0
ro_rang = 0

for base, variants in sorted(grouped_names.items()):
    if len(variants) > 1:
        print(f"⚠️ [NGHI VẤN] Gốc '{base}' có {len(variants)} biến thể: {', '.join(variants)}")
        nghi_van += 1
    else:
        print(f"✅ [RÕ RÀNG] {variants[0]}")
        ro_rang += 1

print("\n" + "="*70)
print(f"TỔNG KẾT: Có {ro_rang} người rõ ràng, và {nghi_van} cụm tên đang bị trùng/nhái.")
print("=> CHỈ THỊ: Cậu hãy nhìn vào các cụm [NGHI VẤN], tự đếm tay lại xem tổng cộng có bao nhiêu người thật, rồi chốt con số đi!")