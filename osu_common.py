import numpy as np

TOKEN_TYPES=['none','circle','slider_start','slider_end','slider_reverse','spinner_start','spinner_end']

SPEED_TYPES=['still',
             'slowest',
             'slower',
             'slow',
             'milder',
             'mild',
             'fast',
             'faster',
             'rapid',
             'fly']

SPEED_BOUND=np.array([1e-6,
             0.1,
             0.18,
             0.32,
             0.56,
             1.0,
             1.8,
             3.16,
             5.62])

class IDX_HIT:
    POS=0
    TYPE=1
    POSITION_X=2
    POSITION_Y=3
    TC_INDEX=4
    BEAT_ID=5
    QUANTIZED_DIST=6
    QUANTIZED_UNIT=7
    SPEED=8

class IDX_KEYPOINT:
    FRAME_ID=0
    HIT_TYPE=1
    POSITION_X=2
    POSITION_Y=3
    SPEED=4
    STATE=5
    BEAT_LEVEL=6

class ENUM_TOKEN:
    NONE=0
    CIRCLE=1
    SLIDER_START=2
    SLIDER_END=3
    SLIDER_REVERSE=4
    SPINNER_START=5
    SPINNER_END=6

class ENUM_CURRENT_STATE:
    NONE=0
    SLIDER=1
    SPINNER=2