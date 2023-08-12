import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import queue
import threading
import pyautogui as pag
import math
# import pygetwindow as gw


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

images = queue.Queue()
mPos = queue.Queue()
isRunning = True


prevPos = (0,0)
prevTime_ms = 0
prevClickTime_ms = 0

TIME_DIFF_ALLOWED = 2 # mostly 1 millisecond
MOUSE_SENSITIVITY = 500
SMOOTHING = 0.015
MOUSE_CLICK_DIST = 0.08
CLICK_DELAY_MS = 10

pag.FAILSAFE = False

MODEL_PATH = 'models/hand_landmarker.task'
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
SCREEN_SIZE = pag.size()
WND_NAME = "video"
# pag.FAILSAFE = True
pag.PAUSE = 0

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

  return annotated_image


def showImage():
    global images, isRunning, WND_NAME

    while isRunning:
        img = images.get()
        if img is not None:
            cv.imshow(WND_NAME, img)
            cv.setWindowProperty(WND_NAME,cv.WND_PROP_TOPMOST,1)

        if cv.waitKey(1) == ord('q'):
            print("we are done")
            cv.destroyAllWindows()
            isRunning = False
            

displayThread = threading.Thread(target=showImage)
displayThread.start()


"""
# def moveMouse():
#     global pos, prevPos
#     lim = 2
#     print("we are getting called")
#     print(pag.position())
#     while isRunning:
#         # print('mouse')
#         pos = mPos.get()
#         # print(pos)
#         if pos is not None:
#             if prevPos is None:
#                 prevPos = pag.position()
#             dif = pos[0] - prevPos[0], pos[1] - prevPos[1]
            
#             print("diff",dif)
#             movePosX, movePosY = 0,0
#             if abs(dif[0]) > lim:
#                 # movPos[0] = dif[0]
#                 movePosX = dif[0]
#             if abs(dif[1]) > lim:
#                 # movPos[1] = dif[1]
#                 movePosY = dif[1]
#         #     # pag.move()
#             movPos = movePosX, movePosY
#             pag.moveRel(movPos[0],movPos[1],_pause = False)

#             # pag.move(dif[0],dif[1],_pause=False)
            
            
#             prevPos = pos


#         # pag.moveTo(pos[0],pos[1],_pause = False)

# mouseThread = threading.Thread(target=moveMouse)
# mouseThread.start()
    # print(pos)
"""
def updatePast(x,y,time_ms):
    global prevPos, prevTime_ms,SMOOTHING
    prevTime_ms = time_ms
    # prevPos = x,y
    prevPos = prevPos[0] if abs(x-prevPos[0]) < SMOOTHING else x, prevPos[1] if abs(y-prevPos[1]) < SMOOTHING else y 
    # prevPos = prevPos[0] if x == 0 else x, prevPos[1] if y == 0 else y 


def moveMyMouse(x,y,time_ms):
    global prevPos, prevTime_ms, SMOOTHING, MOUSE_SENSITIVITY, SCREEN_SIZE

    # if abs(x - prevPos[0]) < SMOOTHING:
    #     x = 0
    # if abs(y - prevPos[1]) < SMOOTHING:
    #     y = 0
    x = round(x, 2)
    y = round(y, 2)
    # print(x,y)
    if (time_ms - prevTime_ms) > TIME_DIFF_ALLOWED:
        # prevTime_ms = time_ms
        # prevPos = x, y
        updatePast(x,y,time_ms)
        return

    # print(x - prevPos[0],y - prevPos[1])

    mov = [(x - prevPos[0]) , (y - prevPos[1])]
    mov[0] = 0 if abs(mov[0]) < SMOOTHING else mov[0]
    mov[1] = 0 if abs(mov[1]) < SMOOTHING else mov[1]
    
    # print(f"mov {mov} || pos {x,y} || prev {prevPos}")
    # pag.moveTo(SCREEN_SIZE[0],SCREEN_SIZE[1])
    c_pos = pag.position()
    mov = mov[0] * MOUSE_SENSITIVITY, mov[1] * MOUSE_SENSITIVITY
    pos = mov[0]  + c_pos[0], mov[1] +c_pos[1]
    pos = max(pos[0],0), max(pos[1],0)
    pos = min(pos[0], SCREEN_SIZE[0] - 2), min(pos[1], SCREEN_SIZE[1]-2)
    # print(pos)

    pag.moveTo(pos[0],pos[1], _pause=False)
    
    
    # updating past values
    # prevTime_ms = time_ms
    # prevPos = x, y
    updatePast(x,y,time_ms)
    return mov

def findFingersUp(result: HandLandmarkerResult, base):
    """
    To find if a is up or not

    Params:
        fingerPoints (HandLandmarkerResult)

        base (int): [fingerBaseIndex] which finger to check if it is up or not
        
        1 thumb, 5 index, 9 middle, 12 ring, 17 little

    """
    fingersPoints = result.hand_landmarks[0]
    # print(fingersPoints[0])
    tip = fingersPoints[base + 3].x, fingersPoints[base + 3].y 
    origin = fingersPoints[base + 1].x, fingersPoints[base + 1].y
    wrist = fingersPoints[0].x, fingersPoints[0].y

    # print(fingerPoints)
    if math.dist(tip,wrist) > math.dist(wrist,origin):
        return True
    return False

def mouseClick(indexTip, middleTip, time_ms, dragOffset = (0,0)):
    global MOUSE_CLICK_DIST, CLICK_DELAY_MS, prevClickTime_ms
    dist = math.dist((indexTip.x, indexTip.y), (middleTip.x,middleTip.y))
    dist = round(dist, 2)
    print(dist)
    if dist >= MOUSE_CLICK_DIST:
        return
    print(f"time diff {time_ms - prevClickTime_ms}")
    if (time_ms - prevClickTime_ms) > CLICK_DELAY_MS:
        pag.click()
        print("mouse click")
    # else:
        # p = pag.position()
        # pag.dragTo(p.x, p.y, _pause = False)
        # print("drag")
    prevClickTime_ms = time_ms

"""
# def mouseClick(indexTip, middleTip, time_ms):
#     global MOUSE_CLICK_DIST, CLICK_DELAY_MS, prevClickTime_ms
#     if (time_ms - prevClickTime_ms) < CLICK_DELAY_MS:
#         return
    
#     dist = math.dist((indexTip.x, indexTip.y), (middleTip.x,middleTip.y))
#     dist = round(dist, 2)
#     print(dist)
#     # pag.click()
    
#     if dist < MOUSE_CLICK_DIST:
#         pag.click()
#         print("mouse click")
#         prevClickTime_ms = time_ms
"""

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print('hand landmarker result: {}'.format(result.landmarks))
    global isRunning,mPos
    images.put(draw_landmarks_on_image(cv.cvtColor(output_image.numpy_view(),cv.COLOR_RGB2BGR),result))

    if len(result.hand_landmarks) < 1:
        return    
    
    index_tip = result.hand_landmarks[0][8]
    middle_tip = result.hand_landmarks[0][12]
    # print(index_tip.z)
    # smooth = 
    # print(findFingersUp(result,5))
    if findFingersUp(result,5):
        # mouse moving offset
        mOff = moveMyMouse(index_tip.x,index_tip.y,timestamp_ms)
        
        if findFingersUp(result,9):
            mouseClick(index_tip,middle_tip, timestamp_ms, mOff)

    # pos = abs(round(index_tip.x,2)*SCREEN_SIZE[0]), abs(round(index_tip.y,2) * SCREEN_SIZE[1])
    # mPos.put(pos)
    # pag.moveTo(pos[0],pos[1],_pause = False)
    # mPos.put((index_tip.x * 50,index_tip.y * 50))
    # print("adding")
    # moveMouse(index_tip.x, index_tip.y)


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with HandLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
    
    vid = cv.VideoCapture(0)
    i = 0
    # cv.namedWindow(WND_NAME,cv.WINDOW_NORMAL)
    isRunning = vid.isOpened()
    while vid.isOpened() and isRunning:
        i += 1
        success, frame = vid.read()
        if not success:
            break
        
        #processing frame 
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.flip(frame, 1)
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)

        landmarker.detect_async(mp_image, i)

        # cv.imshow('video',frame)

        # if cv.waitKey(1) == ord('q'):
        #     break

    vid.release()

cv.destroyAllWindows()
# mouseThread.kill()