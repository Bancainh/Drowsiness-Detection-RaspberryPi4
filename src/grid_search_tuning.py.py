import pandas as pd
import numpy as np
import itertools
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =====================================================================
# 1. KHU VỰC THAM SỐ ĐẦU VÀO VÀ KHÔNG GIAN TÌM KIẾM (GRID SEARCH)
# =====================================================================

FILE_PATH = r'D:\Drowsiness_Project\03_Final_Dataset ( File gộp dữ liệu cuối cùng )\ULTIMATE_MASTER_DATASET.csv'
OUT_REPORT = r'D:\Drowsiness_Project\03_Final_Dataset ( File gộp dữ liệu cuối cùng )\GridSearch_Tuning_Results.csv'

# --- A. NHÓM THAM SỐ CỐ ĐỊNH ---
FPS = 18.9                    
EAR_NOISE_THRESHOLD = 0.05     

# --- B. KHÔNG GIAN TÌM KIẾM (Hãy thêm/bớt các giá trị bạn muốn thử) ---
# Khuyến cáo: Không nên để quá nhiều giá trị trong lần chạy đầu tiên để tránh tốn thời gian
GRID_DROWSY_TIME_SEC = [0.8, 1.0, 1.2, 1.5]     
GRID_SENSITIVITY_FACTOR = [0.75, 0.8, 0.85]     
GRID_CALIB_TIME_SEC = [5.0, 10.0]               
GRID_FFILL_LIMIT = [3, 5]                       

# =====================================================================
# 2. HÀM ĐÁNH GIÁ CHO 1 BỘ THAM SỐ
# =====================================================================
def evaluate_params(df_original, drowsy_time, sensitivity, calib_time, ffill_limit):
    test_df = df_original.copy()
    test_df['Pred_State'] = 'SAFE'
    
    n_calib_frames = int(calib_time * FPS)
    t_drowsy_frames = int(drowsy_time * FPS)
    
    for (user, condition, video), group in test_df.groupby(['User_Name', 'Condition', 'Original_File_x']):
        group_ear = group['EAR_Value'].ffill(limit=ffill_limit)
        calib_raw = group_ear.dropna().head(n_calib_frames)
        calib_filtered = calib_raw[calib_raw > EAR_NOISE_THRESHOLD]
        
        if len(calib_filtered) < (n_calib_frames * 0.3): 
            continue # Bỏ qua các video quá ngắn không đủ hiệu chuẩn
            
        mu_baseline = calib_filtered.mean()
        th_adapt = mu_baseline * sensitivity
        
        is_low_ear = (group_ear < th_adapt).astype(int)
        drowsy_windows = is_low_ear.rolling(window=t_drowsy_frames).sum() >= t_drowsy_frames
        
        drowsy_indices = group.index[drowsy_windows]
        test_df.loc[drowsy_indices, 'Pred_State'] = 'DROWSY'

    # Đánh giá tổng quan (Global Metrics)
    eval_df = test_df.dropna(subset=['Label_GT', 'EAR_Value'])
    if len(eval_df) == 0:
        return 0, 0, 0, 0
        
    y_true_all = eval_df['Label_GT'].apply(lambda x: 1 if str(x).strip().upper() == 'DROWSY' else 0)
    y_pred_all = eval_df['Pred_State'].apply(lambda x: 1 if str(x).strip().upper() == 'DROWSY' else 0)
    
    acc = accuracy_score(y_true_all, y_pred_all)
    prec = precision_score(y_true_all, y_pred_all, zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
    
    return acc, prec, rec, f1

# =====================================================================
# 3. HÀM CHẠY GRID SEARCH
# =====================================================================
def run_grid_search():
    print(f"Đang đọc dữ liệu gốc từ: {FILE_PATH}...")
    try:
        df_original = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file dữ liệu!")
        return

    # Sắp xếp 1 lần duy nhất ở ngoài cùng
    df_original = df_original.sort_values(by=['User_Name', 'Condition', 'Original_File_x', 'Frame_ID'])
    
    # Tạo tất cả các tổ hợp tham số
    param_combinations = list(itertools.product(
        GRID_DROWSY_TIME_SEC, 
        GRID_SENSITIVITY_FACTOR, 
        GRID_CALIB_TIME_SEC, 
        GRID_FFILL_LIMIT
    ))
    
    total_iterations = len(param_combinations)
    print(f"\n=> Bắt đầu chạy thử {total_iterations} tổ hợp tham số khác nhau...")
    
    results_log = []
    start_time = time.time()

    for idx, (d_time, s_factor, c_time, f_limit) in enumerate(param_combinations, 1):
        print(f"[{idx}/{total_iterations}] Đang thử: DrowsyTime={d_time}s, Sens={s_factor}, Calib={c_time}s, Ffill={f_limit}...", end=" ")
        
        acc, prec, rec, f1 = evaluate_params(df_original, d_time, s_factor, c_time, f_limit)
        
        print(f"-> Recall: {rec*100:.1f}%, F1: {f1*100:.1f}%")
        
        results_log.append({
            'DROWSY_TIME_SEC': d_time,
            'SENSITIVITY_FACTOR': s_factor,
            'CALIB_TIME_SEC': c_time,
            'FFILL_LIMIT': f_limit,
            'Recall': rec,
            'F1_Score': f1,
            'Precision': prec,
            'Accuracy': acc
        })

    # Chuyển list kết quả thành DataFrame
    results_df = pd.DataFrame(results_log)
    
    # ƯU TIÊN SẮP XẾP: 1. Recall giảm dần -> 2. F1-Score giảm dần (để tránh trường hợp Recall cao nhưng Precision nát bét)
    results_df = results_df.sort_values(by=['Recall', 'F1_Score'], ascending=[False, False])
    
    print("\n" + "="*50)
    print("🏆 TOP 5 BỘ THAM SỐ TỐT NHẤT (Ưu tiên Recall)")
    print("="*50)
    print(results_df.head(5).to_string(index=False))
    
    # Xuất toàn bộ kết quả ra file
    results_df.to_csv(OUT_REPORT, index=False)
    
    elapsed_time = (time.time() - start_time) / 60
    print(f"\n✅ Đã hoàn thành trong {elapsed_time:.1f} phút!")
    print(f"📊 Chi tiết toàn bộ các lần chạy đã được lưu tại: {OUT_REPORT}")

if __name__ == '__main__':
    run_grid_search()