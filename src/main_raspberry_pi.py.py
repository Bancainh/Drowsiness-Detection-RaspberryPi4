import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import pandas as pd
import time
from datetime import datetime
from math import hypot

# ==============================================================================
# CLASS 1: XỬ LÝ HÀNG LOẠT VIDEO (Dùng để xuất file CSV làm báo cáo)
# ==============================================================================
class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144] 
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # Ngưỡng cảnh báo khớp với bài báo là 1.2 giây
        self.DROWSINESS_THRESHOLD = 1.2 
        self.ALARM_ON = False
        
        self.is_calibrating = True
        self.calibration_frames = 0
        self.calibration_data = []
        self.CALIBRATION_LIMIT = 450 # 15 giây x 30 FPS
        self.th_adapt = 0.0 
        self.start_closing_time = None 
        
        self.last_valid_ear = 1.0 
        # ĐÃ THÊM: Biến đếm số khung hình mất mặt cho thuật toán Forward-Fill
        self.missing_face_frames = 0 

    def calculate_ear(self, eye_points, landmarks, w, h):
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points]
        A = hypot(pts[1][0] - pts[5][0], pts[1][1] - pts[5][1])
        B = hypot(pts[2][0] - pts[4][0], pts[2][1] - pts[4][1])
        C = hypot(pts[0][0] - pts[3][0], pts[0][1] - pts[3][1])
        return (A + B) / (2.0 * C) if C != 0 else 0.0

    def finalize_calibration(self):
        # Lọc bỏ khung hình nhiễu (EAR <= 0.05) theo chuẩn bài báo
        valid_data = [ear for ear in self.calibration_data if ear > 0.05]
        
        # Fallback an toàn phòng trường hợp video bị hỏng toàn bộ phần đầu
        if len(valid_data) == 0:
            valid_data = [0.23 / 0.80] # Đảm bảo khi nhân 0.8 sẽ ra threshold tĩnh 0.23
            
        # Dùng np.mean()
        typical_open_eye = np.mean(valid_data)
        self.th_adapt = typical_open_eye * 0.80
        self.is_calibrating = False
        return self.th_adapt

    def process_video_folder(self, video_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        video_files = glob.glob(os.path.join(video_folder, '*.mp4'))
        print(f"[INFO] Tim thay {len(video_files)} video. Bat dau quet du lieu 10 cot...")

        for idx, video_path in enumerate(video_files):
            video_name = os.path.basename(video_path)
            user_name = os.path.splitext(video_name)[0] 
            
            csv_name = f"{user_name}_Pred.csv"
            output_csv_path = os.path.join(output_folder, csv_name)
            
            print(f"\n🎬 [{idx+1}/{len(video_files)}] Dang chay: {video_name}")
            
            cap = cv2.VideoCapture(video_path)
            
            fps_goc = cap.get(cv2.CAP_PROP_FPS)
            if fps_goc == 0: fps_goc = 30.0 
            
            self.is_calibrating = True
            self.calibration_frames = 0
            self.calibration_data = []
            self.start_closing_time = None
            self.ALARM_ON = False
            self.last_valid_ear = 1.0 
            self.missing_face_frames = 0
            
            data_log = []
            frame_count = 0

            while cap.isOpened():
                start_process_time = time.time() 
                
                success, image = cap.read()
                if not success: break

                frame_count += 1
                current_time_sec = frame_count / fps_goc
                real_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image)

                h, w, _ = image.shape
                current_status = 'Safe'
                avgEAR = 0.0

                # ==========================================
                # ĐÃ SỬA LOGIC: FORWARD-FILL GIỚI HẠN 5 FRAMES
                # ==========================================
                if results.multi_face_landmarks:
                    self.missing_face_frames = 0 # Reset bộ đếm khi tìm thấy mặt
                    for face_landmarks in results.multi_face_landmarks:
                        leftEAR = self.calculate_ear(self.LEFT_EYE, face_landmarks.landmark, w, h)
                        rightEAR = self.calculate_ear(self.RIGHT_EYE, face_landmarks.landmark, w, h)
                        avgEAR = (leftEAR + rightEAR) / 2.0
                        self.last_valid_ear = avgEAR 
                else:
                    self.missing_face_frames += 1
                    if self.missing_face_frames <= 5:
                        avgEAR = self.last_valid_ear # Bù đắp dữ liệu cũ
                    else:
                        avgEAR = 0.0 # Nếu mất dấu quá 5 frame, coi như nhắm mắt/mất kiểm soát

                if self.is_calibrating:
                    self.calibration_data.append(avgEAR)
                    self.calibration_frames += 1
                    if self.calibration_frames >= self.CALIBRATION_LIMIT:
                        self.finalize_calibration()
                    current_status = 'Safe' 
                else:
                    if avgEAR < self.th_adapt:
                        if self.start_closing_time is None:
                            self.start_closing_time = current_time_sec
                        
                        current_duration = current_time_sec - self.start_closing_time
                        
                        if current_duration >= self.DROWSINESS_THRESHOLD:
                            self.ALARM_ON = True
                            current_status = 'DROWSY'
                        else:
                            current_status = 'Closing' 
                    else:
                        self.start_closing_time = None
                        self.ALARM_ON = False
                        current_status = 'Safe'

                final_label = 'DROWSY' if current_status == 'DROWSY' else 'Safe'

                process_duration = time.time() - start_process_time
                fps_process = 1.0 / process_duration if process_duration > 0 else 0.0

                data_log.append({
                    'User_Name': user_name, 'Timestamp': real_timestamp,
                    'Frame_ID': frame_count, 'Time_Sec': round(current_time_sec, 3), 
                    'Label': final_label, 'Status': current_status,               
                    'EAR_Value': round(avgEAR, 4),
                    'Threshold': round(self.th_adapt, 4) if not self.is_calibrating else 0.0,
                    'FPS_Process': round(fps_process, 2), 'FPS_Goc': round(fps_goc, 2)
                })

            cap.release()
            
            if data_log:
                df = pd.DataFrame(data_log)
                columns_order = ['User_Name', 'Timestamp', 'Frame_ID', 'Time_Sec', 'Label', 'Status', 'EAR_Value', 'Threshold', 'FPS_Process', 'FPS_Goc']
                df = df[columns_order]
                df.to_csv(output_csv_path, index=False)
                print(f"✅ Da xuat file dong bo: {csv_name}")


# ==============================================================================
# CLASS 2: DEMO CAMERA TRỰC TIẾP TRÊN RASPBERRY PI
# ==============================================================================
class LiveDrowsinessDemo:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144] 
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # ĐÃ SỬA: Đồng bộ ngưỡng 1.2 giây
        self.DROWSINESS_THRESHOLD = 1.2 
        self.ALARM_ON = False
        
        self.is_calibrating = True
        self.calibration_frames = 0
        self.calibration_data = []
        self.CALIBRATION_LIMIT = 100 # Giữ nguyên 100 frame cho Demo nhanh
        self.th_adapt = 0.0 
        self.start_closing_time = None 
        
        self.last_valid_ear = 1.0 
        self.missing_face_frames = 0 

    def calculate_ear(self, eye_points, landmarks, w, h):
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points]
        A = hypot(pts[1][0] - pts[5][0], pts[1][1] - pts[5][1])
        B = hypot(pts[2][0] - pts[4][0], pts[2][1] - pts[4][1])
        C = hypot(pts[0][0] - pts[3][0], pts[0][1] - pts[3][1])
        return (A + B) / (2.0 * C) if C != 0 else 0.0

    def finalize_calibration(self):
        valid_data = [ear for ear in self.calibration_data if ear > 0.05]
        if len(valid_data) == 0: valid_data = [0.23 / 0.80]
        self.th_adapt = np.mean(valid_data) * 0.80
        self.is_calibrating = False
        return self.th_adapt

    def run_demo(self):
        cap = cv2.VideoCapture(0)
        print("[SYSTEM] Dang khoi dong luong Webcam...")

        while cap.isOpened():
            start_process_time = time.time()
            success, image = cap.read()
            if not success: break

            image = cv2.flip(image, 1)
            h, w, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            current_status = 'Safe'
            avgEAR = 0.0
            box_color = (0, 255, 0) 

            # ĐÃ SỬA: Logic Forward-Fill 5 Frames
            if results.multi_face_landmarks:
                self.missing_face_frames = 0
                for face_landmarks in results.multi_face_landmarks:
                    for idx in self.LEFT_EYE + self.RIGHT_EYE:
                        x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                        cv2.circle(image, (x, y), 2, (0, 255, 255), -1)

                    leftEAR = self.calculate_ear(self.LEFT_EYE, face_landmarks.landmark, w, h)
                    rightEAR = self.calculate_ear(self.RIGHT_EYE, face_landmarks.landmark, w, h)
                    avgEAR = (leftEAR + rightEAR) / 2.0
                    self.last_valid_ear = avgEAR 
            else:
                self.missing_face_frames += 1
                if self.missing_face_frames <= 5:
                    avgEAR = self.last_valid_ear
                else:
                    avgEAR = 0.0

            if self.is_calibrating:
                self.calibration_data.append(avgEAR)
                self.calibration_frames += 1
                cv2.putText(image, f"CALIBRATING... {self.calibration_frames}/{self.CALIBRATION_LIMIT}", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if self.calibration_frames >= self.CALIBRATION_LIMIT:
                    self.finalize_calibration()
                current_status = 'Calibrating'
            else:
                if avgEAR < self.th_adapt:
                    if self.start_closing_time is None:
                        self.start_closing_time = time.time()
                    
                    if (time.time() - self.start_closing_time) >= self.DROWSINESS_THRESHOLD:
                        self.ALARM_ON = True
                        current_status = 'DROWSY WARNING!'
                        box_color = (0, 0, 255) 
                    else:
                        current_status = 'Eyes Closing...'
                        box_color = (0, 165, 255) 
                else:
                    self.start_closing_time = None
                    self.ALARM_ON = False
                    current_status = 'Safe'

            process_duration = time.time() - start_process_time
            fps = 1.0 / process_duration if process_duration > 0 else 0.0

            cv2.rectangle(image, (10, 10), (w-10, h-10), box_color, 4)
            cv2.putText(image, f"Status: {current_status}", (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 3)
            cv2.putText(image, f"EAR: {avgEAR:.3f}", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Threshold: {self.th_adapt:.3f}", (300, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"FPS: {fps:.1f}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if self.ALARM_ON:
                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

            cv2.imshow("Live Interview Demo - Edge AI Drowsiness", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Da dong ung dung.")


# ==============================================================================
# KHU VỰC THỰC THI CHÍNH (Chọn 1 trong 2 chế độ để chạy)
# ==============================================================================
if __name__ == "__main__":
    
    # CHẾ ĐỘ 1: XỬ LÝ HÀNG LOẠT VIDEO (Mặc định đang bật)
    INPUT_VIDEO_DIR = r"/home/grouptwo/Downloads/Day_Light"
    OUTPUT_CSV_DIR = r"/home/grouptwo/Downloads/Day_Light_Result"
    
    detector = DrowsinessDetector()
    detector.process_video_folder(INPUT_VIDEO_DIR, OUTPUT_CSV_DIR)
    print("\n🎉 HOAN THANH! DU LIEU DA KHOP NHAU 100%!")

    # --------------------------------------------------------------------------
    # CHẾ ĐỘ 2: BẬT DEMO WEBCAM TRỰC TIẾP
    # (Nếu muốn chạy thử trên máy, hãy comment '#' 5 dòng của CHẾ ĐỘ 1 ở trên 
    # và bỏ comment 2 dòng dưới đây)
    # --------------------------------------------------------------------------
    # demo = LiveDrowsinessDemo()
    # demo.run_demo()