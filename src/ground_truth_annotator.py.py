import cv2
import pandas as pd
import os
import glob

# --- CẤU HÌNH ---
script_dir = os.path.dirname(os.path.abspath(__file__))
VIDEO_FOLDER = r'D:\Drowsiness_Project\04_Scripts ( Code )'

# --- KIỂM TRA ---
if not os.path.exists(VIDEO_FOLDER):
    print(f"❌ LỖI: Không tìm thấy thư mục 'vid' tại: {VIDEO_FOLDER}")
    exit()

video_files = glob.glob(os.path.join(VIDEO_FOLDER, '*.mp4'))
if not video_files:
    print(f"⚠️ Không tìm thấy file .mp4 nào trong 'vid'.")
    exit()

print(f"✅ Tìm thấy {len(video_files)} video.")
print("-" * 64)
print("HƯỚNG DẪN ĐIỀU KHIỂN (AUTO SKIP 15S):")
print(" [1] : Đánh dấu NGỦ (Drowsy)")
print(" [0] : Đánh dấu TỈNH (Safe)")
print(" [f] : Tăng tốc (x2, x4, x8)")
print(" [n] : Tốc độ thường (x1)")
print(" [b] : TUA LẠI 5 GIÂY (Không tua quá mốc 15s)")
print(" [q] : Qua video tiếp theo")
print("-" * 64)

for idx, video_path in enumerate(video_files):
    video_name = os.path.basename(video_path)
    output_csv_name = video_name.replace('.mp4', '.csv')
    output_path = os.path.join(VIDEO_FOLDER, output_csv_name)
    
    if os.path.exists(output_path):
        print(f"⏩ Đã có file {output_csv_name} -> Bỏ qua.")
        continue

    print(f"\n🎬 [{idx+1}/{len(video_files)}] Đang làm: {video_name}")
    
    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video == 0: fps_video = 30 # Fallback
    
    data = []
    
    # =========================================================
    # LOGIC 1: TỰ ĐỘNG GHI 15S ĐẦU VÀO DỮ LIỆU & SKIP VIDEO
    # =========================================================
    SKIP_SECONDS = 15.0
    frames_to_skip = int(SKIP_SECONDS * fps_video)
    
    # Bơm dữ liệu 15s đầu vào list (Mặc định là 'Safe')
    for i in range(1, frames_to_skip + 1):
        data.append({
            'Frame_ID': i,
            'Time_Sec': round(i / fps_video, 3),
            'Label': 'Safe'
        })
        
    # Ép video nhảy thẳng đến frame thứ frames_to_skip
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames_to_skip)
    
    # Khởi tạo bộ đếm bắt đầu từ sau 15s
    frame_count = frames_to_skip 
    current_label = 'Safe'
    playback_speed = 1.0 
    
    # =========================================================
    # VÒNG LẶP XỬ LÝ VIDEO TỪ GIÂY 15 TRỞ ĐI
    # =========================================================
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1 
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        wait_time = int(1000 / (fps_video * playback_speed))
        if wait_time < 1: wait_time = 1
        
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q'): 
            break
        elif key == ord('1'): 
            current_label = 'DROWSY'
        elif key == ord('0'): 
            current_label = 'Safe'
        elif key == ord('f'): 
            playback_speed *= 2
            if playback_speed > 8: playback_speed = 8
        elif key == ord('n'): 
            playback_speed = 1.0
        elif key == ord('b'): 
            # LOGIC TUA LẠI 5 GIÂY (Khóa mốc 15s)
            frames_to_rewind = int(fps_video * 5)
            # Không cho phép tua lùi qua mốc 15s đầu tiên
            target_frame = max(frames_to_skip, frame_count - frames_to_rewind)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            frame_count = target_frame
            data = data[:target_frame] # Xóa dữ liệu rác vừa gán nhầm
            current_label = 'Safe'
            playback_speed = 1.0 
            continue 

        # Ghi nhận dữ liệu
        data.append({
            'Frame_ID': frame_count,
            'Time_Sec': round(current_time_sec, 3), 
            'Label': current_label
        })

        # --- HIỂN THỊ ---
        color = (0, 0, 255) if current_label == 'DROWSY' else (0, 255, 0)

        cv2.putText(frame, f"STATUS: {current_label}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 6) 
        cv2.putText(frame, f"STATUS: {current_label}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.putText(frame, f"FRAME: {frame_count}", (20, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"TIME: {current_time_sec:.2f}s", (20, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        speed_str = f"SPEED: x{int(playback_speed)}"
        cv2.putText(frame, speed_str, (20, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.putText(frame, "1:NGU | 0:TINH | f:NHANH | n:THUONG | b:TUA LAI", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Tool Gan Nhan - Academic Standard', frame)

    cap.release()
    
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"✅ Đã lưu chuẩn Ground Truth: {output_csv_name}")
    else:
        print("⚠️ Không có dữ liệu để lưu.")

cv2.destroyAllWindows()
print("\n🎉 XONG HẾT RỒI!")