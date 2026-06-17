import cv2
import numpy as np
import glob
import os

def adjust_gamma(image, gamma=2.5):
    # Tạo bảng tra cứu (LookUp Table) để ép pixel theo hàm mũ Gamma
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # Áp dụng bảng tra cứu vào ảnh
    return cv2.LUT(image, table)

def process_videos(input_dir, output_dir, gamma_value=2.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_files = glob.glob(os.path.join(input_dir, '*.mp4'))
    print(f"[INFO] Bắt đầu phù phép {len(video_files)} video thành ban đêm...")

    for video_path in video_files:
        video_name = os.path.basename(video_path)
        # Thêm tiền tố 'Thieu_sang_' để code Evaluation lúc sau tự nhận diện
        output_path = os.path.join(output_dir, f"Thieu_sang_{video_name}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Kéo sáng bằng thuật toán Gamma
            night_frame = adjust_gamma(frame, gamma=gamma_value)
            
            out.write(night_frame)

        cap.release()
        out.release()
        print(f"✅ Đã xử lý xong: {output_path}")

if __name__ == "__main__":
    # >>> SỬA ĐƯỜNG DẪN THƯ MỤC CỦA CẬU VÀO ĐÂY <<<
    INPUT_FOLDER = r"D:\Drowsiness_Project\01_Raw_Videos\Day_Light_glasses"
    OUTPUT_FOLDER = r"D:\Drowsiness_Project\01_Raw_Videos\Low_Light_glasses"
    
    # Gamma = 2.5 là mức khuyên dùng. Thử tăng lên 3.0 nếu thấy vẫn sáng.
    process_videos(INPUT_FOLDER, OUTPUT_FOLDER, gamma_value=3)