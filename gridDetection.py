import numpy as np
import cv2
import matplotlib.pyplot as plt

COLOR = (0, 0, 255)          # Red color (BGR)
THICKNESS = 3
ROW = 3
COLUMN = 5

def get_grid_detection(display=False):
    img = cv2.imread(r'train9\2019-10-25frame3825.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(gray, 50, 150)
    blurred_image = cv2.medianBlur(edged, 3)

    dilated_img = cv2.dilate(blurred_image, np.ones((15, 15), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(dilated_img, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw on
    img_with_rect = img.copy()
    rect_drawn = False

    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # print(f'Bounding Rectangle: \nX={x}, \nY={y}, \nW={w}, \nH={h}')
        
        # Draw the rectangle (Green color BGR: (0, 0, 255))
        cv2.rectangle(img_with_rect, (x, y), (x + w, y + h), (0, 0, 255), 2)
        rect_drawn = True

        for row in range(1, ROW):
            row_start = (x, y + (row * h) // ROW)
            row_end = (x + w, y + (row * h) // ROW)
            cv2.line(img_with_rect, row_start, row_end, COLOR, THICKNESS, cv2.LINE_AA)

        for col in range(1, COLUMN):
            col_start = (x + (col * w) // COLUMN, y)
            col_end = (x + (col * w) // COLUMN, y + h)
            cv2.line(img_with_rect, col_start, col_end, COLOR, THICKNESS, cv2.LINE_AA)

    # Convert the final image with rectangle to RGB for Matplotlib display
    img_final_rgb = cv2.cvtColor(img_with_rect, cv2.COLOR_BGR2RGB)

    if display and rect_drawn:
        cv2.imshow('Detected Grid', img_final_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        fig, axes = plt.subplots(2, 2, figsize=(10, 5))
        axes[0,0].imshow(edged, cmap='gray')
        axes[0,0].set_title('edged')
        axes[0,0].axis('off')

        axes[0,1].imshow(blurred_image, cmap='gray')
        axes[0,1].set_title('blurred')
        axes[0,1].axis('off')

        axes[1,0].imshow(dilated_img, cmap='gray')
        axes[1,0].set_title('dilation')
        axes[1,0].axis('off')

        axes[1,1].imshow(img_final_rgb, cmap='gray')
        axes[1,1].set_title('final')
        axes[1,1].axis('off')

        plt.tight_layout()
        plt.show()

    return (ROW, COLUMN, x, y, w, h)