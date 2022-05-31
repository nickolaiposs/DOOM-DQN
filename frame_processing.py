import numpy as np
from skimage import transform
from collections import deque


def rgb_to_grey(frame):
    # each cell only has one grey value from 0-255 contrary to RGB which has three
    # Y = 0.2125R + 0.7154G + 0.0721B
    rgb_to_grey = [0.2125, 0.7154, 0.0721]
    frame = np.dot(frame, rgb_to_grey)

    return frame


def preprocess(frame, res, top_crop=0, bottom_crop=0, left_crop=0, right_crop=0):
    # convert to greyscale so we can only work with one color channel
    frame = rgb_to_grey(frame)

    width = frame.shape[1]
    height = frame.shape[0]

    bottom_crop = height - bottom_crop
    left_crop = width - left_crop

    # cropping
    frame = frame[top_crop:bottom_crop, right_crop:left_crop]

    # resizing
    frame = transform.resize(frame, res)

    # transform pixel data to range [0, 1]
    frame = frame / 255.0

    return frame


class HistoryFrames:
    def __init__(self, res, history_length, skip_frame=4):
        self.history_length = history_length
        self.res = res

        # for only appending every xth frame, default is 4th
        self.skip_frame = skip_frame
        self.frame_count = 0

        # making blank frames as an initial starting point, and initilizing the deque and history
        self.empty_state = np.zeros(res)
        self.frames, self.history = None, None
        self.reset()

    def next(self, frame):
        # append only xth frame
        self.frame_count += 1
        if not self.frame_count % self.skip_frame == 1:
            return

        # update deque
        self.frames.appendleft(frame)
        self.frames.pop()
        self._update_state()

    def reset(self):
        self.frames = deque(
            [self.empty_state for i in range(self.history_length)])
        self._update_state()

    def _update_state(self):
        # outputs a four dimensional array (res X 4) of the stacked frames
        # the first "axis" is the amount of samples to be inputted into the model, which is 1
        self.history = np.expand_dims(np.stack(self.frames, axis=2), axis=0)

    def state(self):
        return self.history
