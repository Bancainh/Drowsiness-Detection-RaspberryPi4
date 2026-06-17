import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu đối chiếu theo đúng Table 1 trong bài báo khoa học
labels = ['Thanh [9]\n(RPi 3)', 'Vadlamudi [10]\n(RPi 4)', 'Sharma [11]\n(RPi 4)', 'Ming [12]\n(Xavier)', 'OURs\n(RPi 4)']
fps_values = [5.0, 0.86, 30.0, 34.0, 18.9]  # (Chú ý: Thanh [9] báo cáo ghi "Low", giả định biểu diễn là 5.0 để vẽ cột)
acc_values = [86.0, 93.0, 85.0, 95.0, 86.95]

x = np.arange(len(labels))
width = 0.35  

fig, ax1 = plt.subplots(figsize=(10, 6))

# Vẽ cột FPS (Trục Y bên trái)
rects1 = ax1.bar(x - width/2, fps_values, width, label='FPS (Speed)', color='#348ABD')
ax1.set_ylabel('Frames Per Second (FPS)', color='#348ABD', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#348ABD')
ax1.set_ylim(0, 35)

# Khởi tạo trục Y thứ 2 (bên phải) cho Accuracy
ax2 = ax1.twinx()  
rects2 = ax2.bar(x + width/2, acc_values, width, label='Accuracy (%)', color='#E24A33')
ax2.set_ylabel('Accuracy (%)', color='#E24A33', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#E24A33')
ax2.set_ylim(0, 105)

# Cài đặt trục X
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)

# Gắn số liệu trực tiếp lên đỉnh cột
def autolabel(rects, ax, suffix=''):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}{suffix}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1, ax1)
autolabel(rects2, ax2, '%')

# Trang trí IEEE style
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9), fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.title('Performance Benchmarking Against Related Works', fontsize=14, fontweight='bold', pad=20)

fig.tight_layout()
out_path = r'D:\Drowsiness_Project\03_Final_Dataset ( File gộp dữ liệu cuối cùng )\Figure5_Benchmark.png'
plt.savefig(out_path, dpi=300)
print(f"✅ VẼ XONG! Ảnh nét căng tại: {out_path}")
plt.show()