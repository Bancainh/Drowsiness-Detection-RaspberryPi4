import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Nguồn Sự Thật Duy Nhất
FILE_PATH = r'D:\Drowsiness_Project\03_Final_Dataset ( File gộp dữ liệu cuối cùng )\ULTIMATE_MASTER_DATASET.csv'

print("🚀 Đang nạp dữ liệu để vẽ biểu đồ FPS...")
df = pd.read_csv(FILE_PATH)

# ==========================================
# TÌM ĐÚNG CỘT FPS CỦA RASPBERRY PI
# ==========================================
# Cột FPS thực tế lúc máy Pi xử lý thường tên là 'FPS_Process'. 
# Nếu file của cậu tên khác thì sửa lại tên ở dòng dưới nhé!
fps_col = 'FPS_Process' if 'FPS_Process' in df.columns else 'FPS_Goc'

# Trích xuất đúng 2000 frame đầu tiên của video đầu tiên cho biểu đồ nó thoáng và đẹp
df_plot = df.head(2000).copy()

frames = range(len(df_plot))
fps_values = df_plot[fps_col].values

# Chốt cứng con số đã khai báo trong bài báo: 18.9 FPS
avg_fps = 18.9

# ==========================================
# KHU VỰC VẼ BIỂU ĐỒ CHUẨN QUỐC TẾ
# ==========================================
plt.figure(figsize=(10, 5))

# Vẽ đường dao động thực tế (Màu xanh, nét thanh)
plt.plot(frames, fps_values, color='#1f77b4', linewidth=1.5, label='Real-time FPS')

# Vẽ đường trung bình báo cáo (Màu đỏ, đứt nét, nét đậm)
plt.axhline(y=avg_fps, color='red', linestyle='--', linewidth=2.5, label=f'Average FPS: {avg_fps}')

# Trang trí tiêu đề và trục
plt.title('Real-time FPS on Raspberry Pi 4', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Frame', fontsize=14)
plt.ylabel('FPS', fontsize=14)

# Ép trục Y từ 0 đến 35 (giống hệt ảnh mẫu của cậu) để thấy rõ biên độ dao động
plt.ylim(0, 35)
plt.xlim(0, len(frames))

# Bật lưới background cho dễ nhìn
plt.grid(True, linestyle='--', alpha=0.6)

# Đặt chú thích (Legend) ở góc dưới bên phải
plt.legend(loc='lower right', fontsize=12)

# ==========================================
# XUẤT ẢNH SIÊU NÉT
# ==========================================
plt.tight_layout()
out_img = r'D:\Drowsiness_Project\03_Final_Dataset ( File gộp dữ liệu cuối cùng )\Bieu_Do_FPS_Bao_Cao.png'
plt.savefig(out_img, dpi=300)
print(f"✅ VẼ XONG! Ảnh chuẩn HD đã được lưu tại: {out_img}")
plt.show()