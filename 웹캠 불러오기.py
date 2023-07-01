import cv2

videofile = cv2.VideoCapture(0)

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

videofile.release()#
cv2.destroyAllWindows()