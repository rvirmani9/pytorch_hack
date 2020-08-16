import cv2
import numpy as np
import torch
import torchvision
import RealTimeSudokuSolver

def showImage(img, name, width, height):
    new_image = np.copy(img)
    new_image = cv2.resize(new_image, (width, height))
    cv2.imshow(name, new_image)

input_shape = (28, 28, 1)
num_classes = 9

model = torch.load('model_mnist.pt')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture("testvid.mp4")
cap.set(3, 1280)
cap.set(4, 720)
old_sudoku = None
while(True):
    ret, frame = cap.read()
    if ret == True:
        sudoku_frame = RealTimeSudokuSolver.recognize_and_solve_sudoku(frame, model, old_sudoku)
        print(sudoku_frame)
        showImage(sudoku_frame, "AR Sudoku Solver", 1066, 600)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
    else:
        break
        
cap.release()
cv2.destroyAllWindows()