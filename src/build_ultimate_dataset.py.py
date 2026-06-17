import pandas as pd
import os

def clean_filename(name):
    # Ép chuỗi, chuyển thành chữ thường để đập tan mọi phân biệt Hoa/Thường
    name = str(name).lower()
    # Chém bay các đuôi và tiền tố nhiễu
    name = name.replace('.csv', '')
    name = name.replace('_pred', '')
    name = name.replace('thieu_sang_', '')
    return name.strip()

def build_ultimate_dataset(base_dir, output_csv):
    pairs = [
        ("Day_Light_GT.csv", "Day_Light_Pred.csv", "Day_Light"),
        ("Day_Light_Glasses_GT.csv", "Day_Light_Glasses_Pred.csv", "Day_Light_Glasses"),
        ("Night_Light_GT.csv", "Night_Light_Pred.csv", "Night_Light"),
        ("Night_Light_Glasses_GT.csv", "Night_Light_Glasses_Pred.csv", "Night_Light_Glasses")
    ]

    all_data = []
    print("[INFO] Bắt đầu gộp Ultimate Master Dataset...")

    for gt_name, pred_name, condition in pairs:
        gt_path = os.path.join(base_dir, gt_name)
        pred_path = os.path.join(base_dir, pred_name)

        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"❌ MẤT TÍCH: Không tìm thấy file ở điều kiện {condition}.")
            continue

        print(f"-> Đang xử lý cặp: {condition}...")
        df_gt = pd.read_csv(gt_path)
        df_pred = pd.read_csv(pred_path)

        # ============================================================
        # ÉP KHUÔN CHÌA KHÓA: DÙNG HÀM CLEAN_FILENAME VÀ ÉP KIỂU INT
        # ============================================================
        df_gt['Join_Key'] = df_gt['Original_File'].apply(clean_filename)
        df_pred['Join_Key'] = df_pred['Original_File'].apply(clean_filename)
        
        df_gt['Frame_ID'] = df_gt['Frame_ID'].astype(int)
        df_pred['Frame_ID'] = df_pred['Frame_ID'].astype(int)

        # Giao nhau lấy phần chung, vứt frame rác
        df_merged = pd.merge(df_gt, df_pred, on=['Join_Key', 'Frame_ID'], how='inner')

        # Dọn tên cột cho chuẩn mực
        label_gt_col = 'Label_x' if 'Label_x' in df_merged.columns else 'Label'
        label_pred_col = 'Label_y' if 'Label_y' in df_merged.columns else ('Label_Pred' if 'Label_Pred' in df_merged.columns else 'Label')
        df_merged.rename(columns={label_gt_col: 'Label_GT', label_pred_col: 'Label_Pred'}, inplace=True)

        # Đóng dấu trường hợp môi trường
        df_merged['Condition'] = condition
        
        # Chỉ gom vào nếu df_merged có dữ liệu
        if len(df_merged) > 0:
            all_data.append(df_merged)
            print(f"   + Ghép thành công: {len(df_merged):,} frames.")
        else:
            print(f"   ⚠️ CẢNH BÁO: Cặp {condition} sau khi ghép bị rỗng 0 frame. Kiểm tra lại dữ liệu bên trong!")

    if not all_data:
        print("\n❌ LỖI: Dữ liệu vẫn trống không. Hai bên không nhận diện được nhau!")
        return

    # Gộp 4 khối dữ liệu
    ultimate_df = pd.concat(all_data, axis=0, ignore_index=True)
    
    # Dọn dẹp cột Join_Key thừa thãi
    if 'Join_Key' in ultimate_df.columns:
        ultimate_df.drop(columns=['Join_Key'], inplace=True)
        
    cols = ultimate_df.columns.tolist()
    if 'Condition' in cols:
        cols.insert(0, cols.pop(cols.index('Condition')))
        ultimate_df = ultimate_df[cols]

    ultimate_df.to_csv(output_csv, index=False)
    
    print("="*60)
    print("✅ ĐÃ ĐÚC THÀNH CÔNG TẬP DỮ LIỆU TỐI THƯỢNG!")
    print(f"📂 Lưu tại: {output_csv}")
    print(f"📊 Tổng số Frame thực chiến thu được: {len(ultimate_df):,}")
    print("="*60)

if __name__ == "__main__":
    BASE_DIR = r"D:\Drowsiness_Project\03_Final_Dataset ( File gộp dữ liệu cuối cùng )" 
    
    # Ép đường dẫn Output vào đúng thư mục đó
    OUTPUT_FILE = r"D:\Drowsiness_Project\03_Final_Dataset ( File gộp dữ liệu cuối cùng )\ULTIMATE_MASTER_DATASET.csv"
    build_ultimate_dataset(BASE_DIR, OUTPUT_FILE)
