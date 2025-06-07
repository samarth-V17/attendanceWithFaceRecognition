import csv
import cv2
from tkinter import*
from PIL import ImageTk, Image
import face_recognition
from datetime import datetime
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials


# Define the scope and authenticate with Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Use the downloaded JSON file here
creds = ServiceAccountCredentials.from_json_keyfile_name("btech7thsem-2153cd0f3f36.json", scope)
client = gspread.authorize(creds)

# Open the Google Sheet by name or URL
sheet = client.open("Attendance 7th SEM Project").sheet1  # Sheet1 is the default sheet


def make_entry(date, name, rollno, status):
    next_row = len(sheet.get_all_values()) + 1  # `get_all_values` returns a list of all filled rows
    sheet.update_cell(next_row, 1, date)   # Add date in the first column
    sheet.update_cell(next_row, 2, name)   # Add name in the second column
    sheet.update_cell(next_row, 3, rollno)  # Add rollno in the third column
    sheet.update_cell(next_row, 4, status)  # Add status in the fourth column



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# function to mark attendance using webcam via opencv
def atten():
    video_capture = cv2.VideoCapture(0)

    # loading face into face_recognition for encoding
    tiwari_image = face_recognition.load_image_file("photos/tiwari.jpg")
    tiwari_encoding = face_recognition.face_encodings(tiwari_image)[0]

    chandu_image = face_recognition.load_image_file("photos/chandu.jpg")
    chandu_encoding = face_recognition.face_encodings(chandu_image)[0]

    sachin_image = face_recognition.load_image_file("photos/sachin.jpg")
    sachin_encoding = face_recognition.face_encodings(sachin_image)[0]
    
    samarth_image = face_recognition.load_image_file("photos/samarth.jpg")
    samarth_encoding = face_recognition.face_encodings(samarth_image)[0]
    
    
    known_face_encoding = [
    tiwari_encoding,
    samarth_encoding,sachin_encoding,chandu_encoding
    ]
    
    known_faces_details = {
    "Abhishek Tiwari":"100210103","Samarth Varshney":"100210150","Sachidanand Yadav":"100210148","Chandrabhan Yadav":"100210124"
    }

    known_faces_names = list(known_faces_details.keys())
    
    students = known_faces_names.copy()
    
    face_locations = []
    face_encodings = []
    face_names = []
    s=True
    
    # using date-time module to find out the current date and time
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")  
    
    # starting the webcam to capture live feed
    while True:
        _,frame = video_capture.read()
        frame = cv2.resize(frame, (640, 480))  # Reduce resolution
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Allowing multiple face detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 6)

    # Creating Rectangle around face
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 250), 2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

   # matching the encoded image with the detected face
        small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        rgb_small_frame = small_frame[:,:,::-1]
        if s:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
                name=""
                face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]
    
                face_names.append(name)
                if name in known_faces_names:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10,100)
                    fontScale              = 1
                    fontColor              = (0,255,0)
                    thickness              = 2
                    lineType               = 2
    
                # to display the name of the identified student
                    cv2.putText(frame,name, 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
    
                    if name in students:
                        students.remove(name)
                        make_entry(current_date,name,known_faces_details[name],"Present")
                        current_time = now.strftime("%H-%M-%S")
                        
        cv2.imshow("attendence system",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    
    

atten()




