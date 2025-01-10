from typing import List
import numpy as np
import imageio
import cv2
import copy
import glob
import os

def load_images(filenames: List) -> List:
    return [imageio.imread(filename) for filename in filenames]


def show_image(image):
    cv2.imshow("Image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def write_image(img,filename):
    cv2.imwrite(filename,img)

def get_chessboard_points(chessboard_shape, dx, dy):
    nx = chessboard_shape[0]
    ny = chessboard_shape[1]
    n = nx*ny # número de puntos que hay
    chessboard_points = np.zeros((n,3), dtype=np.float32)
    for y in range(ny):
        for x in range(nx):
            N = x+nx*y # para cada y que pasamos le sumamos los 8 puntos
            chessboard_points[N][0] = x*dx
            chessboard_points[N][1] = y*dy
    return chessboard_points


if __name__ == "__main__":
    imgs_path = sorted(glob.glob("../data/FotosCalibracion/*png"))
    imgs = load_images(imgs_path)

    pattern_size = (7, 7)
    corners = [cv2.findChessboardCorners(image, pattern_size) for image in imgs]

    corners_copy = copy.deepcopy(corners)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    # TODO To refine corner detections with cv2.cornerSubPix() you need to input grayscale images. Build a list containing grayscale images.
    imgs_gray = [cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) for image in imgs]

    corners_refined = [cv2.cornerSubPix(i, cor[1], (7,7), (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]

    imgs_copy = copy.deepcopy(imgs)
    corners_to_draw = [cv2.drawChessboardCorners(img,(7,7),np.array(corners),len(corners)>0) for img,corners in zip(imgs_copy,corners_refined)]

    os.makedirs("../data/cornersAjedrez",exist_ok=True)
    for i,img in enumerate(corners_to_draw):
        show_image(img)
        write_image(img, f"../data/cornersAjedrez/corners_ajedrez_{str(i).zfill(3)}.jpg")

    chessboard_shape = (7,7)
    dx = 30
    dy = 30
    get_chessboard_points(chessboard_shape,dx,dy)

    chessboard_points = [get_chessboard_points((7, 7), 30, 30) for img in imgs[1:]]

    valid_corners = [cor[1] for cor in corners if cor[0]]
    # Convert list to numpy array
    valid_corners = np.asarray(valid_corners, dtype=np.float32)

    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chessboard_points,valid_corners,(8,6),imgs_gray[0].shape[::-1],None,None)

    # Obtain extrinsics
    extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))
    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Root mean squared reprojection error:\n", rms)