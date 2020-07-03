import cv2

def serialize_kp(kp):
    return {
        'angle': kp.angle, 
        'class_id': kp.class_id,
        'octave': kp.octave,
        'pt': kp.pt,
        'response': kp.response,
        'size': kp.size
    }

def deserialize_kp(d):
    return cv2.KeyPoint(d['pt'][0], d['pt'][1], d['size'], d['angle'], d['response'], d['octave'], d['class_id'])