import math

def calculate_distance(p1, p2, image_width, image_height):
    x1, y1 = int(p1.x * image_width), int(p1.y * image_height)
    x2, y2 = int(p2.x * image_width), int(p2.y * image_height)
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return int(distance), (x1, y1), (x2, y2)
