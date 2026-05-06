import os
import cv2
import numpy as np
from YuNetFace import FaceDetectorYunet
import pandas as pd

fd = FaceDetectorYunet()

# Folder to save the collected face images
dataset_folder = "dataset"
os.makedirs(dataset_folder, exist_ok=True)

student_name = input("Enter Student Name or ID: ")
student_folder = os.path.join(dataset_folder, student_name)

# Create student-specific folder
if not os.path.exists(student_folder):
    os.makedirs(student_folder)

def start_webcam():
    rtsp_username = "admin"
    # rtsp_password = "123456789"
    rtsp_password = "cctv@123"
    width = 800
    height = 480
    cam_no = "1"
    rtsp = "rtsp://" + rtsp_username + ":" + rtsp_password + "@192.168.1.64:554/Streaming/channels/" + cam_no + "01"
    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
    # cap.open(rtsp)
    cap.set(3, width)  # Set width
    cap.set(4, height)  # Set height
    # success, current_cam = cap.read()
    return cap

# Initialize webcam (0 for default camera)
# cap = start_webcam()
cap = cv2.VideoCapture(0)

# Counter to keep track of image number
image_counter = 0

print("Press 'q' to quit or 'c' to capture faces.")

while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = fd.detect(frame)
        # if faces:
        #     fd.draw_faces(frame, faces)
        # show_image(frame)
        if not faces:
            continue
            
        if len(faces) > 0:  # Check if faces are detected
            fd.draw_faces(frame, faces)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                for face in faces:
                    # try:
                    #     x, y, w, h = map(int, face[:4])  # Extract face bounding box
                    # except TypeError:
                    #     print(f"Skipping invalid face data: {face}")
                    #     continue
                    try:
                        x1, y1 = int(face["x1"]), int(face["y1"])
                        x2, y2 = int(face["x2"]), int(face["y2"])
                        w, h = x2 - x1, y2 - y1 +10 # Compute width and height
                    except KeyError:
                        print(f"Skipping invalid face data: {face}")
                        continue 
                    
                    face_roi = frame[y1:y1+h, x1:x1+w]  # Crop face from frame
                    
                    if face_roi.size == 0:
                        continue  # Skip empty ROIs (in case of bad detection)
                    
                    face_filename = os.path.join(student_folder, f"face_{image_counter}.jpg")
                    # faces = np.array(faces)
                    cv2.imwrite(face_filename, face_roi)
                    image_counter += 1
                    print(f"Face captured and saved as {face_filename}")
                    if image_counter >= 5:
                        break 
                    
        cv2.imshow('YuNet Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # elif key == ord('c'):
    #     print("Press 'c' to capture face or 'q' to quit")

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

students = pd.read_csv("students_marked.csv")
new_row = {
    "Roll_No": student_name.split(" ")[0],
    "Students": student_name.split(" ")[1],
    "Attendance": "Absent"
}

students = pd.concat([students, pd.DataFrame([new_row])], ignore_index=True)
# students["Roll_No"].append(student_name.split(" ")[0])
# students["Name"].append(student_name.split(" ")[1])
students.to_csv("students_marked.csv", index=False)
os.system("python create_embeddings.py")