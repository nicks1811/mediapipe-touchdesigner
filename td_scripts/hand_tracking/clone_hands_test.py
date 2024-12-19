import cv2
import mediapipe as mp

mp_hand = mp.solutions.hands
hands = mp_hand.Hands()

mp_drawing_utils = mp.solutions.drawing_utils

cap = cv2.VideoCapture(2)

while True:
    success, img = cap.read()

    if not success:
        break

    result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            mp_drawing_utils.draw_landmarks(
                                            img,
                                            hand_landmark,
                                            mp_hand.HAND_CONNECTIONS
                                            )
                                            
                                            
                                            
                
                           
    

    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()