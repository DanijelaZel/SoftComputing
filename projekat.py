import cv2
import numpy as np
import time
from keras.models import load_model
from scipy import ndimage
from distancaBrojLinija import distance, pnt2line

model = load_model('profModel.h5')

def HoughTransformacija(img):

    G = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Threshold1 = 50
    Threshold2 = 150
    E = cv2.Canny(G, Threshold1, Threshold2, apertureSize=3)
    lines = cv2.HoughLinesP(E, rho=1, theta=1 * np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)

    for x1, y1, x2, y2 in lines[0]:
        x = x1
        y = y1
        xx = x2
        yy = y2

    i = 0
    while i < len(lines):
        for x1, y1, x2, y2 in lines[i]:
            if x1 < x:
                x = x1
                y = y1
            if x2 > xx:
                yy = y2
                xx = x2
            i = i + 1

            # plt.figure()
            # plt.imshow(I)
            # plt.show()
            # plt.title('Hough Lines')
            #cv2.line(I, (x1, y1), (x2, y2), (255, 0, 0), 3)
            #cv2.imwrite('pronadjenaLinija.jpg', I)

    return x, y, xx, yy

def jednacinaPrave(x1,y1):
    global x
    global y
    global xx
    global yy
    jednacina = (xx - x)*(y1 - y) - (yy - y)*(x1 - x)
    return jednacina

cc = -1
def nextId():
    global cc
    cc += 1
    return cc

def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal

sum = 0;

def pracenjeBrojeva(img2):
    global sum

    kernel = np.ones((2, 2), np.uint8)
    lower = np.array([230, 230, 230])
    upper = np.array([255, 255, 255])

    mask = cv2.inRange(img2, lower, upper)
    img0 = 1.0 * mask

    img0 = cv2.dilate(img0, kernel)  # cv2.erode(img0,kernel)
    img0 = cv2.dilate(img0, kernel)

    x, y = el['center']

    image = img0[y - 14: y + 14, x - 14: x + 14]
    resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_NEAREST)
    skaliraj = resized / 255
    vektor = skaliraj.flatten()
    kolona = np.reshape(vektor, (1, 784))
    slikaNM = np.array(kolona, dtype=np.float32)
    nmb = model.predict(slikaNM)

    sum += np.argmax(nmb)
    print "Trenutna suma: " + str(sum)

for brVidea in range(0,10):

    nazivVidea = "video/video-" + str(brVidea) + ".avi"
    cap = cv2.VideoCapture(nazivVidea)
    print nazivVidea

    kernel = np.ones((2, 2), np.uint8)
    lower = np.array([230, 230, 230])
    upper = np.array([255, 255, 255])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter('video/video-' + str(brVidea) + '-output.avi', fourcc, 20.0, (640, 480))

    sum = 0
    elements = []
    t =0
    times = []
    ret = True
    prviFrejmVidea = True

    while (ret):

        start_time = time.time()
        ret, img = cap.read()
        img2=img
        if (prviFrejmVidea):
            prviFrejmVidea = False
            x1, y1, x2, y2 = HoughTransformacija(img)
            line = [(x1, y1), (x2, y2)]
            print("Linija")
            print(line)


        if (ret == True):
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(img, lower, upper)
            img0 = 1.0 * mask

            img0 = cv2.dilate(img0, kernel)
            img0 = cv2.dilate(img0, kernel)

            labeled, nr_objects = ndimage.label(img0)
            objects = ndimage.find_objects(labeled)
            for i in range(nr_objects):
                loc = objects[i]
                (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                            (loc[0].stop + loc[0].start) / 2)
                (dxc, dyc) = ((loc[1].stop - loc[1].start),
                              (loc[0].stop - loc[0].start))

                if (dxc > 11 or dyc > 11):

                    elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
                    # find in range
                    lst = inRange(20, elem, elements)
                    nn = len(lst)
                    if nn == 0:
                        elem['id'] = nextId()
                        elem['t'] = t
                        elem['pass'] = False
                        elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
                        elem['future'] = []
                        elements.append(elem)
                    elif nn == 1:
                        lst[0]['center'] = elem['center']
                        lst[0]['t'] = t
                        lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})
                        lst[0]['future'] = []

            for el in elements:
                tt = t - el['t']
                if (tt < 3):
                    dist, pnt, r = pnt2line(el['center'], line[0], line[1])
                    if r > 0:
                        cv2.line(img, pnt, el['center'], (0, 255, 25), 1)
                        c = (25, 25, 255)
                        if (dist < 9):
                            c = (0, 255, 160)
                            if el['pass'] == False:
                                el['pass'] = True
                                pracenjeBrojeva(img2)
                    cv2.circle(img, el['center'], 16, c, 2)

                    id = el['id']
                    cv2.putText(img, str(el['id']),
                                (el['center'][0] + 10, el['center'][1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                   # for hist in el['history']:
                        #ttt = t - hist['t']
                       # if (ttt < 100):
                           # cv2.circle(img, hist['center'], 1, (0, 255, 255), 1)

                    for fu in el['future']:
                        ttt = fu[0] - t
                        if (ttt < 100):
                            cv2.circle(img, (fu[1], fu[2]), 1, (255, 255, 0), 1)

            elapsed_time = time.time() - start_time
            times.append(elapsed_time * 1000)

            cv2.putText(img, 'Suma brojeva: ' + str(sum), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            t += 1

            cv2.imshow(nazivVidea, img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            out.write(img)
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    et = np.array(times)
    print "Ukupna suma " + nazivVidea + " je: "+str(sum)

