import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = 'chessboard.mp4'
K = np.array([[1735.925, 0, 993.063],
              [0, 1735.934, 573.203],
              [0, 0, 1]])
dist_coeff = np.array([0.251413, -1.188235, 0.009189, 0.011738, 1.854149])
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Prepare a 3D triangle for simple AR
triangle_lower = board_cellsize * np.array([[4, 2, 0], [6, 2, 0], [5, 4, 0]]) # 바닥 삼각형
triangle_upper = board_cellsize * np.array([[4, 2, -1], [6, 2, -1], [5, 4, -1]]) # 상단 삼각형

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the triangle on the image
        line_lower, _ = cv.projectPoints(triangle_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv.projectPoints(triangle_upper, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(line_lower)], True, (255, 128, 64), 2) # 하늘색
        cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2) # 빨간색
        for b, t in zip(line_lower, line_upper):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2) # 초록색

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('AR Object Visualization', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()
