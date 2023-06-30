import face_recognition
import cv2
import numpy as np

#카메라 연결. input을 통한 카메라 연결은 추가 필요. 본인 얼굴 인식 프로그램
video_capture = cv2.VideoCapture(0) #0은 인덱스값(카메라 번호)

#샘플을 불러와 인식하는 클래스
class Sample:
    def __init__(self, image_path):
        self.image_path = image_path
    
    #image_path 경로에 있는 사진을 불러와 image에 저장한다.
    def load_image(self):
        self.image = face_recognition.load_image_file(self.image_path) 
    
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
    
#dongyoung_image = face_recognition.load_image_file("dongyoung.jpg")
#dongyoung_face_encoding = face_recognition.face_encodings(dongyoung_image)[0] #위의 카메라 인덱스와는 별개의 인코딩값. 이것도 인식하는 사진이 많아질수록 숫자가 늘어남.
















#여기는 그냥 공식 문서 보면서 처음에 적어뒀던 것들. 참고하면서 작성함.

#사진에서 얼굴 찾기
image = face_recognition.load_image_file("caimage.jpg") #임의로 정한 사진 이름
face_locations = face_recognition.face_locations(image)

#사진에서 얼굴의 특징을 찾기
#import face_recognition
image=face_recognition.load_image_file("camimage.jpg")

#사진 속 얼굴의 신원 확인하기
#import face_recognition
known_image = face_recognition.load_image_file("camimage.jpg") #이미 확인된 얼굴사진 (미리 학습이 필요.)
unknown_image = face_recognition.load_image_file("unknowncamimage.jpg") #앞으로 확인할(인식할) 얼굴사진

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)