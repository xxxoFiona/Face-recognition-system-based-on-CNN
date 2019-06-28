import tensorflow as tf

import  cv2
import  random
import  matplotlib.pyplot as plt
import  numpy as np
#keep_prob_ = graph.get_tensor_by_name("keep_prob:0")
class Camera_reader(object):
    #在初始化camera的时候建立模型，并加载已经训练好的模型

 def build_camera(self):
    # opencv文件中人脸级联文件的位置，用于帮助识别图像或者视频流中的人脸
    face_cascade = cv2.CascadeClassifier('F:\python\haarcascade_frontalface_alt.xml')
    # 打开摄像头并开始读取画面
    cameraCapture = cv2.VideoCapture(0)
    success, frame = cameraCapture.read()
    #a=['female','male']
    while success and cv2.waitKey(1) == -1:
        success, frame = cameraCapture.read()
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)  # 识别人脸

        for (x, y, w, h) in faces:
            ROI = frame[y:y + h,x:x + w]
            ROI1 = cv2.resize(ROI, (92, 112), interpolation=cv2.INTER_LINEAR)
            grayimg=cv2.cvtColor(ROI1, cv2.COLOR_BGR2GRAY)

            img0=cv2.cvtColor(grayimg,cv2.COLOR_GRAY2BGR)
            img = img0.astype('float32')
            # img = img/255.0
            img2 = img.reshape((-1, 30912))
            #img = np.reshape(img,[-1,112,91,3])
            y_ = sess.run(op_to_predict, feed_dict={input:img2})
            label = y_[0] # 找出概率最高的
            #cv2.imwrite("face.jpg", img)
            if label == 0:
                show_name = 'male'
            else:
                show_name = 'female'
            #cv2.putText(frame, show_name, (x, y +20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  # 显示名字
            #frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sp = imghat.shape
            hatSizeH = int(sp[0] / sp[1] * w)
            if hatSizeH > y:
                hatSizeH = y
            hatResize = cv2.resize(imghat, (w, hatSizeH), interpolation=cv2.INTER_NEAREST)
            dilateResize = cv2.resize(dilate, (w, hatSizeH), interpolation=cv2.INTER_NEAREST)
            top = y - hatSizeH
            if top <= 0:
                top = 0
            rows, cols, channels = hatResize.shape
            #roi = img[top:top + rows, x:x + cols]

           #for i in range(rows):
               #for j in range(cols):
                    #if dilateResize[i, j] == 0:  # 0代表黑色的点
                        #frame[top + i, x + j] = hatResize[i, j]  # 此处替换颜色，为BGR通道
                        #cv2.imwrite("hat4.jpg", hatResize)

            c = cv2.waitKey(33) & 0xFF
            if c == ord('q'):
                cv2.destroyAllWindows()
                break;
            elif c == ord('r'):
                cv2.imwrite("hat.jpg",img0)
            elif c == ord('a'):
                cv2.imwrite("1.jpg",ROI1)
            elif c == ord('b'):
                cv2.imwrite("2.jpg",frame)
            elif c == ord('s'):
                cv2.imwrite("hatadd.jpg", hatResize)
        cv2.imshow("Camera", frame)

    cameraCapture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sess = tf.Session()
   # model = os.path.abspath('model/')
    saver = tf.train.import_meta_graph('model3/my-gender-v1.0.meta')
    saver.restore(sess, tf.train.latest_checkpoint('model3/'))
    graph = tf.get_default_graph()
    op_to_predict = graph.get_tensor_by_name('op_to_predict:0')
    #y = tf.get_collection('network-output')
    input= graph.get_tensor_by_name('input_images:0')
    imghat = cv2.imread('hat2.jpg')
    # 构建mask
    hsv = cv2.cvtColor(imghat, cv2.COLOR_RGB2HSV)
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 46])
    mask = cv2.inRange(hsv, black_lower, black_upper)
    cv2.imwrite('mask.jpg',mask)
    # 腐蚀膨胀
    erode = cv2.erode(mask, None, iterations=1)
    cv2.imwrite('erode.jpg',erode)
    dilate = cv2.dilate(erode, None, iterations=1)
    cv2.imwrite("dalate.jpg", dilate)
    # cv2.imshow('dilate',dilate)
    camera = Camera_reader()
    camera.build_camera()