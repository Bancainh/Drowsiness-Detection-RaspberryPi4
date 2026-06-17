import pandas as pd
import glob
import os

def merge_folder_safely(input_folder, output_csv):
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"❌ Trống rỗng! Không tìm thấy file CSV nào ở: {input_folder}")
        return

    print(f"[INFO] Bắt đầu gộp {len(csv_files)} file từ: {os.path.basename(input_folder)}")
    
    df_list = []
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            
            # BƯỚC SỐNG CÒN: Đóng dấu tên file gốc vào dữ liệu
            filename = os.path.basename(filepath)
            df['Original_File'] = filename 
            
            # Đưa cột Original_File lên đầu cho dễ nhìn
            cols = df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            df = df[cols]
            
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ Lỗi đọc file {filepath}: {e}")

    # Xếp chồng theo chiều dọc (UNION ALL)
    master_df = pd.concat(df_list, axis=0, ignore_index=True)
    
    # Xuất file với đường dẫn tuyệt đối
    master_df.to_csv(output_csv, index=False)
    
    print("="*50)
    print(f"✅ GỘP THÀNH CÔNG!")
    print(f"📂 File Master: {output_csv}")
    print(f"📊 Tổng số khung hình (Frames): {len(master_df):,}")
    print("="*50)

if __name__ == "__main__":
    # >>> 1. ĐƯỜNG DẪN THƯ MỤC CẦN GỘP <<<
    # Thay bằng đường dẫn thư mục Ground Truth (hoặc Predicted) của cậu
    INPUT_DIR = r"D:\Drowsiness_Project\01_Raw_Videos\Low_Light_glasses\Low_Light_glassess_Pre" 
    
    # >>> 2. ĐƯỜNG DẪN FILE ĐÍCH (ÉP TUYỆT ĐỐI VÀO Ổ D) <<<
    OUTPUT_FILE = r"D:\Drowsiness_Project\01_Raw_Videos\Low_Light_glasses\Low_Light_glassess_Pre\Night_Light_Glasses_Pred.csv"   
    
    merge_folder_safely(INPUT_DIR, OUTPUT_FILE)