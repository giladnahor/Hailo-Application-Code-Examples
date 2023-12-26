import hailo
import time
# Importing VideoFrame before importing GST is must
from gsthailo import VideoFrame
from gi.repository import Gst

def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def run(video_frame: VideoFrame):
    aspect_ratio = 16/9
    # scale the bbox to fit original aspect ratio
    # the original aspect ratio is 16:9 for example
    # the inferred image aspect ratio is 1:1 with borders on the top and bottom
    #|----------------------|
    #|    (black border     |
    #|                      |
    #|------top_border------|
    #|                      |
    #|     scaled image     |
    #|                      |
    #|----bottom_border-----|
    #|                      |
    #|    (black border     |
    #|----------------------|   
    bottom_border = (1-1/aspect_ratio)/2
    top_border = 1 - bottom_border

    # in addition we want to get square bboxes to prevent distorsion in the cropper
    detections = video_frame.roi.get_objects_typed(hailo.HAILO_DETECTION)
    for detection in detections:
        bbox = detection.get_bbox()
        # lets map y coordinates to the original image
        ymin = map(bbox.ymin(), bottom_border, top_border, 0, 1)
        ymax = map(bbox.ymax(), bottom_border, top_border, 0, 1)
        height = ymax - ymin
        # get required x coordinates
        xmin = bbox.xmin()  
        width = bbox.width()
        
        # lets get make the bbox square (need to take aspect ratio into account)
        normalized_height = height / aspect_ratio
        if normalized_height > width:
            xmin = xmin + (width - height / aspect_ratio)/2
            width = height / aspect_ratio
        elif normalized_height < width:
            ymin = ymin + (height - width * aspect_ratio)/2
            height = width * aspect_ratio
        new_bbox = hailo.HailoBBox(xmin, ymin, width, height)
        detection.set_bbox(new_bbox)

    return Gst.FlowReturn.OK
