import os


def transform_distance(self, x, x_distance, y, y_distance):
    if not self.base_top_left:
        return [x - x_distance, y - y_distance]

    return [x + x_distance, y + y_distance]


def transform_location(self, x, y):
    if not self.base_top_left:
        return [64 - x, 64 - y]

    return [x, y]


def kill_process(name):
    os.system("taskkill /f /im " + name)
