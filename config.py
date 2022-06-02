class Config:
    ROOT_PATH = '.'
    FRAME_HEIGHT = 360
    CLEAR_INPUT_FRAME_CACHE = True
    INPUT_FPS = 20
    INPUT_VIDEO_NAME = 'content.mp4'
    INPUT_VIDEO_PATH = f'{ROOT_PATH}/static/videos/input/{INPUT_VIDEO_NAME}'
    INPUT_FRAME_DIRECTORY = f'{ROOT_PATH}/static/images/input'
    INPUT_FRAME_FILE = '{:0>4d}_frame.png'
    INPUT_FRAME_PATH = f'{INPUT_FRAME_DIRECTORY}/{INPUT_FRAME_FILE}'

    STYLE_REF_DIRECTORY = f'{ROOT_PATH}/static/images/style'
    STYLE_SEQUENCE = [0, 1, 2]

    OUTPUT_FPS = 20
    OUTPUT_VIDEO_NAME = 'output.mp4'
    OUTPUT_VIDEO_PATH = f'{ROOT_PATH}/static/videos/output/{OUTPUT_VIDEO_NAME}'
    OUTPUT_FRAME_DIRECTORY = f'{ROOT_PATH}/static/images/output'
    OUTPUT_FRAME_FILE = '{:0>4d}_frame.png'
    OUTPUT_FRAME_PATH = f'{OUTPUT_FRAME_DIRECTORY}/{OUTPUT_FRAME_FILE}'

    GHOST_FRAME_TRANSPARENCY = 0.1

    TENSORFLOW_CACHE_DIRECTORY = 'models'
    TENSORFLOW_HUB_HANDLE = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
