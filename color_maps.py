import numpy as np

color = {
    0: (0, 0, 0),  # 검 background
    1: (255, 255, 255),  # 흰 cavity t-uco buried missing
    2: (255, 71, 71),  # 빨 sand pebble waste
    3: (255, 196, 71),  # 노 manhole undefined_object inverse_cavity
    4: (71, 255, 89),  # 초 pipe inverse_pipe
    5: (71, 162, 255),  # 파


}

def make_to_rgb(images):
    re_images = np.zeros((3, images.shape[0], images.shape[1]))
    it = np.nditer(images, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        re_images[:, idx[0], idx[1]] = np.array(color[images[idx]]) / 255

        it.iternext()

    return re_images