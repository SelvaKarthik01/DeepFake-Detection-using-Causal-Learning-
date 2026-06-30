import cv2
import mediapipe as mp
import numpy as np
import os 

def extract_golden_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    # Correct way to initialize FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    golden_indices = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        # Skip every 2nd frame for speed (optional)
        if frame_count % 2 != 0: continue 

        # TIER 1: Sharpness Check
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 50: continue 

        # TIER 2: Frontal Pose Check
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Using specific landmark indices: 
            # 1 = Nose Tip, 33 = Left Eye Outer, 263 = Right Eye Outer
            nose_x = landmarks[1].x
            l_eye_x = landmarks[33].x
            r_eye_x = landmarks[263].x
            
            # Calculate symmetry: distance from nose to each eye
            dist_l = abs(nose_x - l_eye_x)
            dist_r = abs(r_eye_x - nose_x)
            
            symmetry_ratio = dist_l / (dist_r + 1e-6)
            
            # Check for Frontal View (Ratio near 1.0)
            if 0.7 < symmetry_ratio < 1.3:
                golden_indices.append(frame_count)
                # Visual feedback (optional)
                print(f"Frame {frame_count} is GOLDEN (Ratio: {symmetry_ratio:.2f})")

    cap.release()
    return golden_indices

def extract_golden_frames(video_path, output_folder="golden_frames_output"):
    # 1. Create the folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    cap = cv2.VideoCapture(video_path)
    golden_indices = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        
        # ... (Insert your Pose/Sharpness logic here) ...
        # Assume 'is_golden' is the result of your symmetry check
        is_golden = True # Placeholder for your logic

        if is_golden:
            golden_indices.append(frame_count)
            # 2. Save the frame to the folder
            file_name = f"frame_{frame_count:04d}.jpg"
            save_path = os.path.join(output_folder, file_name)
            cv2.imwrite(save_path, frame)
            
    cap.release()
    return golden_indices

# Test the function
if __name__ == "__main__":
    # Replace with your actual video path
    video_file = "test_video.mp4" 
    indices = extract_golden_frames(video_file)
    print(f"\nTotal Golden Frames Found: {len(indices)}")
