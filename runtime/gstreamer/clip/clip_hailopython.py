import json
from pathlib import Path

import hailo
import numpy as np

# Importing VideoFrame before importing GST is must
from gsthailo import VideoFrame
from gi.repository import Gst
# import text_image_matcher using singleton_manager to make sure that only one instance of the TextImageMatcher class is created.
from singleton_manager import text_image_matcher

def run(video_frame: VideoFrame):
    top_level_matrix = video_frame.roi.get_objects_typed(hailo.HAILO_MATRIX)
    if len(top_level_matrix) == 0:
        detections = video_frame.roi.get_objects_typed(hailo.HAILO_DETECTION)
    else:
        detections = [video_frame.roi] # Use the ROI as the detection
    
    embeddings_np = None
    used_detection = []
    for detection in detections:
        # TBD relevant only for person detection
        if detection.get_label() != 'person': # Only run on person detections
            continue
        results = detection.get_objects_typed(hailo.HAILO_MATRIX)
        if len(results) == 0:
            print("No matrix found in detection")
            continue
        # Convert the matrix to a NumPy array
        detection_embedings = np.array(results[0].get_data())
        used_detection.append(detection)
        if embeddings_np is None:
            embeddings_np = detection_embedings[np.newaxis, :]
        else:
            embeddings_np = np.vstack((embeddings_np, detection_embedings))

    if embeddings_np is not None:
        matches = text_image_matcher.match(embeddings_np)
        if (text_image_matcher.global_best):
            # remove all classifications
            for detection in detections:
                old_classification = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
                for old in old_classification:
                    detection.remove_object(old)
        for match in matches:
            (row_idx, label, confidence) = match
            detection = used_detection[row_idx]
            old_classification = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
            for old in old_classification:
                detection.remove_object(old)
            # Add label as classification metadata
            classification = hailo.HailoClassification('clip', label, confidence)
            detection.add_object(classification)
    # for detection in detections:
    #     results = detection.get_objects_typed(hailo.HAILO_MATRIX)
    #     if len(results) > 0:
    #         # Convert the matrix to a NumPy array
    #         embeddings_np = np.array(results[0].get_data())
    #         (label, confidence) = text_image_matcher.detect(embeddings_np)
    #         if label is not None:
    #             old_classification = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
    #             for old in old_classification:
    #                 detection.remove_object(old)
    #             # Add label as classification metadata
    #             classification = hailo.HailoClassification('clip', label, confidence)
    #             detection.add_object(classification)
    
    # if debug_launch():
    #     import ipdb; ipdb.set_trace()
    return Gst.FlowReturn.OK
