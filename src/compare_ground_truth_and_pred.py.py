import pandas as pd
import numpy as np
import os

# =============================================================================
# 1. CẤU HÌNH THÔNG SỐ
# =============================================================================
# Lưu ý: Thay đổi đường dẫn cho đúng với máy bạn
FILE_GROUND_TRUTH = r"D:\Detailed_Evaluation_Report.csv"
FILE_PREDICTION   = r"D:\MASTER_DATASET_ALL.csv"

THRESH_RATIO    = 0.8
WAIT_TIME_SEC   = 1.2   
EAR_MIN_VALID   = 0.05   
LOOKBACK_SEC    = 1   
CALIB_FRAMES    = 283    # <--- ĐÃ SỬA: 15s * 18.9 FPS = ~283 frames

GLASSES_USERS = ['bao', 'cuong', 'duong', 'huy', 'khanh', 'quan', 'sangdeokinh', 'son', 'tien', 'ton']

# =============================================================================
# 2. HÀM XỬ LÝ CHÍNH
# =============================================================================

def calculate_metrics():
    print(f"--- ĐANG TÌM FILE ---")
    if not os.path.exists(FILE_GROUND_TRUTH) or not os.path.exists(FILE_PREDICTION):
        print(f"LỖI: Không tìm thấy file csv tại {FILE_GROUND_TRUTH} hoặc {FILE_PREDICTION}")
        return
        
    print(f"Tham số: Ratio={THRESH_RATIO}, Calib={CALIB_FRAMES} frames")
    
    gt_df = pd.read_csv(FILE_GROUND_TRUTH)
    pred_df = pd.read_csv(FILE_PREDICTION)
    
    # Phân loại nhóm
    all_users = gt_df['User_Name'].unique()
    user_group_map = {u: ('Đeo Kính' if u in GLASSES_USERS else 'Không Đeo') for u in all_users}

    processed_preds = []
    
    # ĐÃ SỬA: Ước lượng FPS ~ 18.9 (Khớp với báo cáo RPi 4)
    WAIT_FRAMES = int(WAIT_TIME_SEC * 18.9)
    if WAIT_FRAMES < 1: WAIT_FRAMES = 1

    for user in pred_df['User_Name'].unique():
        user_df = pred_df[pred_df['User_Name'] == user].copy()
        user_df = user_df.sort_values(by='Frame_ID')
        
        # --- LOGIC MỚI: Chỉ lấy 283 frame đầu để tính Baseline ---
        # Lấy 283 dòng đầu tiên
        calib_data = user_df.head(CALIB_FRAMES)
        
        # Chỉ tính trung bình trên những frame hợp lệ (EAR > 0.05)
        valid_calib = calib_data[calib_data['EAR_Value'] > EAR_MIN_VALID]['EAR_Value']
        
        if len(valid_calib) > 10: # Cần ít nhất 10 frame sạch để tin tưởng
            baseline = valid_calib.mean()
        else:
            # ĐÃ SỬA: Fallback về 0.23 chuẩn tĩnh
            baseline = 0.23 
            
        threshold = baseline * THRESH_RATIO
        
        # Áp dụng ngưỡng này cho TOÀN BỘ video
        user_df['Instant_Status'] = np.where(user_df['EAR_Value'] <= EAR_MIN_VALID, np.nan,
                                             np.where(user_df['EAR_Value'] < threshold, 1, 0))
        
        # Xử lý No Face & Wait Time
        user_df['Instant_Status'] = user_df['Instant_Status'].ffill().fillna(0)
        user_df['Simulated_Label'] = user_df['Instant_Status'].rolling(window=WAIT_FRAMES, min_periods=1).min()
        user_df['Pred_Label_Str'] = user_df['Simulated_Label'].apply(lambda x: 'DROWSY' if x == 1 else 'Safe')
        
        processed_preds.append(user_df)
        
    new_pred_df = pd.concat(processed_preds)

    # --- TÍNH TOÁN KẾT QUẢ ---
    gt_df['Time_Sec_Video'] = gt_df['Time_Sec_Video'].astype(float)
    new_pred_df['Time_Sec_Video'] = new_pred_df['Time_Sec_Video'].astype(float)
    new_pred_df = new_pred_df.dropna(subset=['Time_Sec_Video'])
    
    final_stats = []

    for user in all_users:
        g_u = gt_df[gt_df['User_Name'] == user]
        p_u = new_pred_df[new_pred_df['User_Name'] == user]
        if p_u.empty: continue
            
        p_u = p_u.sort_values('Time_Sec_Video')
        p_times = p_u['Time_Sec_Video'].values
        p_labels = p_u['Pred_Label_Str'].values
        
        tp, fn, fp, tn = 0, 0, 0, 0
        
        for _, row in g_u.iterrows():
            t_gt = row['Time_Sec_Video']
            label_gt = row['Label']
            
            # Lookback
            t_start = t_gt - LOOKBACK_SEC
            t_end = t_gt + 0.2
            mask = (p_times >= t_start) & (p_times <= t_end)
            window_preds = p_labels[mask]
            
            machine_final = 'Safe'
            if len(window_preds) > 0 and 'DROWSY' in window_preds:
                machine_final = 'DROWSY'
            
            if label_gt == 'DROWSY':
                if machine_final == 'DROWSY': tp += 1
                else: fn += 1
            else:
                if machine_final == 'DROWSY': fp += 1
                else: tn += 1
        
        final_stats.append({
            'User': user, 'Group': user_group_map.get(user, 'Khác'),
            'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn
        })
        
    stats_df = pd.DataFrame(final_stats)
    
    def get_metrics(df):
        TP, FN, FP, TN = df['TP'].sum(), df['FN'].sum(), df['FP'].sum(), df['TN'].sum()
        acc = (TP + TN) / (TP + TN + FP + FN) if (TP+TN+FP+FN) > 0 else 0
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        return pd.Series({'Precision': round(prec*100,2), 'Recall': round(rec*100,2), 'Accuracy': round(acc*100,2)})

    print("\n" + "="*30)
    print(" KẾT QUẢ (CALIB 283 FRAMES)")
    print("="*30)
    print(stats_df.groupby('Group').apply(get_metrics))

if __name__ == "__main__":
    calculate_metrics()