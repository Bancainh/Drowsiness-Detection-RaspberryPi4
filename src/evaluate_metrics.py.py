import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =====================================================================
# 1. KHU VỰC THAM SỐ ĐẦU VÀO
# =====================================================================

FILE_PATH = r'D:\Drowsiness_Project\03_Final_Dataset ( File gộp dữ liệu cuối cùng )\ULTIMATE_MASTER_DATASET.csv'

# --- A. NHÓM THAM SỐ CỐ ĐỊNH (Phụ thuộc phần cứng & Sinh lý) ---
FPS = 18.90                     
EAR_NOISE_THRESHOLD = 0.05     

# --- B. NHÓM THAM SỐ ĐIỀU CHỈNH (Cần tinh chỉnh để tối ưu Recall/Accuracy) ---

DROWSY_TIME_SEC = 1.2          # Còi hú nếu nhắm mắt liên tục 1.2s (Lọc rác do chớp mắt chậm/ngáp).
SENSITIVITY_FACTOR = 0.80      # Ngưỡng bắt lỗi: Mắt sập qua mốc 80% so với lúc tỉnh thì tính là nhắm.
CALIB_TIME_SEC = 15.0          # Dùng 10s đầu tiên để đo chuẩn mắt lúc mở to nhất làm mốc (Base EAR).
FFILL_LIMIT = 5                # Bù dữ liệu tạm tối đa 3 khung hình nếu camera rớt nhịp do xe xóc.              

# =====================================================================
# 2. KHU VỰC LOGIC XỬ LÝ
# =====================================================================
def run_detailed_evaluation():
    print(f"Đang đọc dữ liệu từ: {FILE_PATH}...")
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file dữ liệu! Vui lòng kiểm tra lại đường dẫn.")
        return

    df = df.sort_values(by=['User_Name', 'Condition', 'Original_File_x', 'Frame_ID'])
    
    test_df = df.copy()
    test_df['Pred_State'] = 'SAFE'
    
    n_calib_frames = int(CALIB_TIME_SEC * FPS)
    t_drowsy_frames = int(DROWSY_TIME_SEC * FPS)
    
    # Tạo một danh sách để lưu kết quả của từng file
    file_results = []

    print("\n--- ĐANG ĐÁNH GIÁ TỪNG FILE. VUI LÒNG CHỜ... ---")

    for (user, condition, video), group in test_df.groupby(['User_Name', 'Condition', 'Original_File_x']):
        
        group['EAR_Value'] = group['EAR_Value'].ffill(limit=FFILL_LIMIT)
        calib_raw = group['EAR_Value'].dropna().head(n_calib_frames)
        calib_filtered = calib_raw[calib_raw > EAR_NOISE_THRESHOLD]
        
        if len(calib_filtered) < (n_calib_frames * 0.3): 
            # Ghi nhận file bị lỗi/quá ngắn không đủ hiệu chuẩn
            file_results.append({
                'User_Name': user, 'Condition': condition, 'File_Name': video,
                'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1_Score': 0,
                'Actual_Drowsy': 0, 'Predicted_Drowsy': 0,
                'Note': 'Lỗi: Không đủ dữ liệu Calibration (Video quá ngắn)'
            })
            continue
            
        mu_baseline = calib_filtered.mean()
        th_adapt = mu_baseline * SENSITIVITY_FACTOR
        
        is_low_ear = (group['EAR_Value'] < th_adapt).astype(int)
        
        drowsy_windows = is_low_ear.rolling(window=t_drowsy_frames).sum() >= t_drowsy_frames
        drowsy_indices = group.index[drowsy_windows]
        
        test_df.loc[drowsy_indices, 'Pred_State'] = 'DROWSY'

        # --- TÍNH ĐIỂM CHO RIÊNG FILE NÀY ---
        eval_group = test_df.loc[group.index].dropna(subset=['Label_GT', 'EAR_Value'])
        
        if len(eval_group) == 0:
            continue
            
        y_true_file = eval_group['Label_GT'].apply(lambda x: 1 if str(x).strip().upper() == 'DROWSY' else 0)
        y_pred_file = eval_group['Pred_State'].apply(lambda x: 1 if str(x).strip().upper() == 'DROWSY' else 0)
        
        acc = accuracy_score(y_true_file, y_pred_file)
        prec = precision_score(y_true_file, y_pred_file, zero_division=0)
        rec = recall_score(y_true_file, y_pred_file, zero_division=0)
        f1 = f1_score(y_true_file, y_pred_file, zero_division=0)
        
        actual_drowsy = sum(y_true_file)
        pred_drowsy = sum(y_pred_file)
        
        # Thêm kết quả vào danh sách
        file_results.append({
            'User_Name': user,
            'Condition': condition,
            'File_Name': video,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'Actual_Drowsy': actual_drowsy,
            'Predicted_Drowsy': pred_drowsy,
            'Note': 'Bình thường' if actual_drowsy > 0 else 'Video KHÔNG CÓ nhãn buồn ngủ'
        })

    # --- ĐÁNH GIÁ TỔNG TOÀN BỘ DATASET ---
    eval_df = test_df.dropna(subset=['Label_GT', 'EAR_Value'])
    y_true_all = eval_df['Label_GT'].apply(lambda x: 1 if str(x).strip().upper() == 'DROWSY' else 0)
    y_pred_all = eval_df['Pred_State'].apply(lambda x: 1 if str(x).strip().upper() == 'DROWSY' else 0)
    
    print("\n====== KẾT QUẢ TỔNG (GLOBAL METRICS) ======")
    print(f"Accuracy  : {accuracy_score(y_true_all, y_pred_all) * 100:.2f}%")
    print(f"Precision : {precision_score(y_true_all, y_pred_all, zero_division=0) * 100:.2f}%")
    print(f"Recall    : {recall_score(y_true_all, y_pred_all, zero_division=0) * 100:.2f}%")
    print(f"F1-Score  : {f1_score(y_true_all, y_pred_all, zero_division=0) * 100:.2f}%")
    print("===========================================")

    # --- XỬ LÝ BÁO CÁO CHI TIẾT ---
    results_df = pd.DataFrame(file_results)
    
    # Chỉ lọc các file CÓ xuất hiện tình trạng buồn ngủ để tìm xem hệ thống bắt trượt ở đâu
    drowsy_videos = results_df[results_df['Actual_Drowsy'] > 0]
    
    # Sắp xếp theo Recall từ Thấp -> Cao (File nào Recall = 0 tức là trượt 100% sẽ nổi lên đầu)
    worst_files = drowsy_videos.sort_values(by=['Recall', 'F1_Score'], ascending=[True, True])
    
    print("\n--- TOP 10 FILE BỊ LỖI HOẶC KẾT QUẢ THẤP NHẤT (CẦN KIỂM TRA LẠI) ---")
    cols_to_show = ['User_Name', 'Condition', 'File_Name', 'Recall', 'Accuracy', 'Actual_Drowsy', 'Predicted_Drowsy']
    print(worst_files.head(10)[cols_to_show].to_string(index=False))

    # Xuất toàn bộ 100 file ra CSV để mở bằng Excel
    out_file = r'D:\Drowsiness_Project\03_Final_Dataset ( File gộp dữ liệu cuối cùng )\Detailed_Evaluation_Report.csv'
    results_df.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f"\n=> Đã xuất chi tiết toàn bộ {len(results_df)} files ra: {out_file}")
    print("Mẹo: Hãy mở file này bằng Excel, dùng Filter để lọc các file có Recall < 0.5 để kiểm tra lại dữ liệu nhé!")

if __name__ == '__main__':
    run_detailed_evaluation()