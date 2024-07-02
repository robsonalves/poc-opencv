import numpy as np
import cv2
import sys
from random import randint


TEXT_COLOR = (randint(0, 255), randint(0,255), randint(0,255))
BORDER_COLOR = (randint(0, 255), randint(0,255), randint(0,255))

FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_SOURCE = 'videos/Traffic_4.mp4'

BGS_TYPES = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']

def get_kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3,3), np.uint8)

    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3,3), np.uint8)

    return kernel

def get_filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, get_kernel('opening'), iterations=2)
    if filter == 'dilation':
        return  cv2.dilate(img, get_kernel('dilation'), iterations=2)
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel('closing'), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, get_kernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, get_kernel('dilation'), iterations=2)
        return dilation


def get_bgsubstractor(BGS_TYPE):
    if BGS_TYPE == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames= 120, decisionThreshold = 0.8)

    if BGS_TYPE == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG(history = 200, nmixtures= 5, backgroundRatio = 0.7, noiseSigma=0)

    if BGS_TYPE == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2(history= 500, detectShadows= True, varThreshold=100)

    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

    if BGS_TYPE == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True, maxPixelStability=15*60,
                                                        isParallel=True)

    print('invalid KERNEL')
    sys.exit(0)


cap = cv2.VideoCapture(VIDEO_SOURCE)
bg_subtractor = get_bgsubstractor(BGS_TYPES[4])
BGS_TYPES = BGS_TYPES[1]

def main():
    while cap.isOpened():
        ok, frame = cap.read()
        # print(ok)
        # print(frame.shape)

        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        bg_mask = bg_subtractor.apply(frame)
        fg_mask = get_filter(bg_mask, 'dilation')
        fg_mask_closing = get_filter(bg_mask, 'closing')
        fg_mask_opening = get_filter(bg_mask, 'opening')
        fg_mask_combine = get_filter(bg_mask, 'combine')

        res = cv2.bitwise_and(frame, frame, mask=fg_mask)
        res_closing = cv2.bitwise_and(frame, frame, mask=fg_mask_closing)
        res_opening = cv2.bitwise_and(frame, frame, mask=fg_mask_opening)
        res_combine = cv2.bitwise_and(frame, frame, mask=fg_mask_combine)

        cv2.putText(res_combine, 'Background subtractor: ' + BGS_TYPES, (10,50), FONT,1, BORDER_COLOR, 3, cv2.LINE_AA)
        if not ok:
            print('End Processing video')
            break

        if BGS_TYPES != 'MOG' and BGS_TYPES != 'GMG':
            cv2.imshow('Background model', bg_subtractor.getBackgroundImage())

        cv2.imshow('Frame', frame)
        cv2.imshow('BG Mask', bg_mask)
        # cv2.imshow('FG Mask', fg_mask)
        cv2.imshow('Dilation Final', res)
        cv2.imshow('Closing Final', res_closing)
        cv2.imshow('Opning Final', res_opening)
        cv2.imshow('Combine Final', res_combine)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()