import face_recognition
import cv2
import numpy as np

#카메라 연결. input을 통한 카메라 연결은 추가 필요. 본인 얼굴 인식 프로그램
videofile = cv2.VideoCapture(0) #0은 인덱스값(카메라 번호)인데 0은 주로 노트북의 카메라

if videofile.isOpened():
    while True:
        vret, img = videofile.read()
        if vret:
            cv2.imshow('webcam', img)
            if cv2.waitKey(1) != -1: #원래는 0인데 영상이라 멈춰있으면 안 돼서 1ms를 기다리도록 설정
                break
        else:
            print("프레임 정상적이지 않음")
            break

else :
    print("오류 발생")

#샘플을 불러와 인식하는 클래스
class SampleImage:
    def __init__(self, image):
        self.image = image

    #image_path 경로에 있는 사진을 불러와 image에 저장한다.
    def load_image(self):
        self.image = face_recognition.load_image_file(self.image)
    
    #image에 저장된 사진에서 얼굴을 인식하고 얼굴 인코딩을 face_encodings에 저장한다.
    #이때 인코딩은 정보가 코드화 된 것.
    def encode_faces(self):
        self.face_encodings = face_recognition.face_encodings(self.image)
    
    #사진에서 불러온 인코딩이(얼굴 정보) 하나가 아닐 수 있지만 첫 번째 인코딩(가장 눈에 띄는 얼굴)을 불러온다. 얼굴이 하나만 나온 사진이 인식률이 높을 것.
    def knownimage_face_encoding(self):
        if self.face_encodings:
            return self.face_encodings[0]
        #사진에서 발견된 얼굴이 없어 인코딩이 없다면 아무것도 추출되지 않을 것.
        else:
            return None
        

a = SampleImage('dongyoung.jpg')

known_face_encodings = [
    a.face_encodings
]

face_locations = [] 
face_encodings = [] # 위에서 작업된 인코딩 리스트
face_names = [] #인코딩마다 붙여진 이름 리스트
process_this_frame = True

while True:
    #Grab a single frame of video
    ret, frame = videofile.read()

    if process_this_frame:
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = a[best_match_index]

            face_names.append(name)

        process_this_frame = not process_this_frame
            
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

videofile.release()
cv2.destroyAllWindows()



