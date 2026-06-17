import pandas as pd

FILE_PATH = r'D:\Drowsiness_Project\03_Final_Dataset ( File gộp dữ liệu cuối cùng )\ULTIMATE_MASTER_DATASET.csv'

print("🔍 Đang đếm số lượng video thực tế được sử dụng trong mỗi trường hợp...")
df = pd.read_csv(FILE_PATH)

# Tìm đúng cột chứa tên file gốc
file_col = 'Original_File_x' if 'Original_File_x' in df.columns else 'Original_File'

# Gom nhóm theo Condition và đếm số tên file phân biệt (duy nhất)
video_counts = df.groupby('Condition')[file_col].nunique()

print("\n" + "="*50)
print("📊 TỔNG KẾT SỐ LƯỢNG VIDEO THEO TRƯỜNG HỢP:")
print("="*50)
tong_video = 0

for condition, count in video_counts.items():
    print(f"- {condition:<25}: {count} video")
    tong_video += count

print("-" * 50)
print(f"👉 TỔNG CỘNG TOÀN BỘ DATASET : {tong_video} video")
print("="*50)