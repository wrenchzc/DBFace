
import common
import cv2

from face import detect, detect_image, HAS_CUDA
from model.DBFace import DBFace


def do_detect_image(model, file):
    objs = detect_image(model, file)
    image = common.imread(file)
    for obj in objs:
        common.drawbbox(image, obj)

    print(objs)
    common.imwrite("detect_result/" + common.file_name_no_suffix(file) + ".draw.jpg", image)


def image_demo():

    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()

    dbface.load("model/dbface.pth")
    do_detect_image(dbface, "datas/selfie.jpg")
    do_detect_image(dbface, "datas/12_Group_Group_12_Group_Group_12_728.jpg")
    do_detect_image(dbface, "datas/head.jpg")
    do_detect_image(dbface, "datas/head2.jpg")
    do_detect_image(dbface, "datas/head3.jpg")
    do_detect_image(dbface, "datas/feichi2.jpeg")


def camera_demo():

    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()

    dbface.load("model/dbface.pth")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()

    while ok:
        objs = detect(dbface, frame)

        for obj in objs:
            common.drawbbox(frame, obj)

        cv2.imshow("demo DBFace", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        ok, frame = cap.read()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_demo()
    #camera_demo()
    


    
