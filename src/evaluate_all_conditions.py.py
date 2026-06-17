import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

FILE_PATH = r'D:\Drowsiness_Project\03_Final_Dataset ( File gộp dữ liệu cuối cùng )\ULTIMATE_MASTER_DATASET.csv'

# THAM SỐ VÀNG ĐÃ CHỐT SỔ
DROWSY_TIME = 1.2
SENSITIVITY = 0.80
CALIB_TIME = 15.0
FFILL_LIMIT = 5
FPS = 18.9
EAR_NOISE = 0.05

print("🚀 1. Đang nạp BẢO VẬT DỮ LIỆU ULTIMATE MASTER...")
df = pd.read_csv(FILE_PATH)

df['Pred_State'] = 'SAFE'
n_calib = int(CALIB_TIME * FPS)
t_drowsy = int(DROWSY_TIME * FPS)

# Linh hoạt tìm cột định danh file để Groupby
file_col = 'Original_File_x' if 'Original_File_x' in df.columns else ('Original_File' if 'Original_File' in df.columns else None)
group_cols = ['Condition']
if 'User_Name' in df.columns: group_cols.append('User_Name')
if file_col: group_cols.append(file_col)

print("🧠 2. Đang kích hoạt Thuật toán EAR để tính toán Nhãn dự đoán...")
for name, group in df.groupby(group_cols):
    ear = group['EAR_Value'].ffill(limit=FFILL_LIMIT)
    calib = ear.dropna().head(n_calib)
    calib_filt = calib[calib > EAR_NOISE]
    if len(calib_filt) < (n_calib * 0.3): continue
        
    th_adapt = calib_filt.mean() * SENSITIVITY
    is_low = (ear < th_adapt).astype(int)
    drowsy_win = is_low.rolling(window=t_drowsy).sum() >= t_drowsy
    df.loc[group.index[drowsy_win], 'Pred_State'] = 'DROWSY'

# Rút trích nhãn chuẩn mực
df_eval = df.dropna(subset=['Label_GT', 'EAR_Value']).copy()
df_eval['y_true'] = df_eval['Label_GT'].apply(lambda x: 1 if str(x).strip().upper() == 'DROWSY' else 0)
df_eval['y_pred'] = df_eval['Pred_State'].apply(lambda x: 1 if str(x).strip().upper() == 'DROWSY' else 0)

print("\n" + "="*70)
print(f"{'ĐIỀU KIỆN (CONDITION)':<25} | {'ACCURACY':<8} | {'PRECISION':<9} | {'RECALL':<8}")
print("="*70)

# ==========================================
# KHU VỰC VẼ BIỂU ĐỒ 4 TRƯỜNG HỢP VÀO 1 ẢNH
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# Dựa vào đúng 4 điều kiện cậu đã gán trong file gộp
conditions = ['Day_Light', 'Day_Light_Glasses', 'Night_Light', 'Night_Light_Glasses']

for i, cond in enumerate(conditions):
    subset = df_eval[df_eval['Condition'] == cond]
    if len(subset) == 0: continue
    
    acc = accuracy_score(subset['y_true'], subset['y_pred']) * 100
    prec = precision_score(subset['y_true'], subset['y_pred'], zero_division=0) * 100
    rec = recall_score(subset['y_true'], subset['y_pred'], zero_division=0) * 100
    print(f"{cond:<25} | {acc:>7.2f}% | {prec:>8.2f}% | {rec:>7.2f}%")
    
    cm = confusion_matrix(subset['y_true'], subset['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                xticklabels=['Safe (Pred)', 'Drowsy (Pred)'], 
                yticklabels=['Safe (True)', 'Drowsy (True)'], annot_kws={"size": 16})
    
    axes[i].set_title(f'{cond}\nAcc: {acc:.1f}% | Rec: {rec:.1f}%', fontsize=16, fontweight='bold')
    axes[i].set_xlabel('Predicted Label', fontsize=14)
    axes[i].set_ylabel('True Label', fontsize=14)

print("="*70)
plt.tight_layout()
out_img = r'D:\Drowsiness_Project\03_Final_Dataset ( File gộp dữ liệu cuối cùng )\Bieu_Do_Bao_Cao_Chuan.png'
plt.savefig(out_img, dpi=300)
print(f"✅ VẼ XONG! Ảnh nét căng tại: {out_img}")
plt.show()