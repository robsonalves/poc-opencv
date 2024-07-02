import numpy as np
import cv2 #openCV

print(cv2.__version__)

VIDEO_SOURCE = 'videos/Cars.mp4'
VIDEO_OUT = 'videos/results/temporal_media_filter.avi'
cap = cv2.VideoCapture(VIDEO_SOURCE)
has_frame, frame = cap.read()
print(has_frame, frame.shape)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, 25, (frame.shape[1],frame.shape[0]), False)

# print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(np.random.uniform(size = 25))
frames_ids = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size = 25)
# print(frames_ids)

# cap.set(cv2.CAP_PROP_POS_FRAMES, 1180)
# has_frame, frame = cap.read()
# cv2.imshow('Test', frame)
# cv2.waitKey(0)

frames = []
for fid in frames_ids:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    has_frame, frame = cap.read()
    frames.append(frame)

# print (np.asarray(frames).shape)
# print(frames[0])

# for frame in frames:
#     cv2.imshow('Frame', frame)
#     cv2.waitKey(0)


print(np.mean([1,3,5,6,8,9]))
print(np.median([[1,3,5,6,8,9]]))

#Background Isolation
median_frame = np.median(frames, axis = 0).astype(dtype=np.uint8)
# print(median_frame)
# cv2.imshow('Median Frame', median_frame)
# cv2.waitKey(0)

cv2.imwrite('model_median_frame.jpg', median_frame)

# GRAY SCALE
cap.set(cv2.CAP_PROP_POS_FRAMES,0)
gray_median_frame = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray', gray_median_frame)
# cv2.waitKey(0)

while(True):
    has_frame, frame = cap.read()

    if not has_frame:
        print('End of the video')
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dframe = cv2.absdiff(frame_gray, gray_median_frame)
    # if Pixel intensity is greater than the set threshold, value set to 255, else set to 0(black)
    # th, dframe = cv2.threshold(dframe, 70, 255,cv2.THRESH_BINARY)
    th, dframe = cv2.threshold(dframe, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(th)
    cv2.imshow('Frame', dframe)
    writer.write(dframe)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()