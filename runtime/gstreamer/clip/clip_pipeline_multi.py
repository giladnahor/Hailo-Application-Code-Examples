import os
import re
batch_size = 8
def get_pipeline_multi(current_path, detector_pipeline, sync, input_uri, tappas_workspace, tapppas_version):
    # Initialize directories and paths
    RESOURCES_DIR = os.path.join(current_path, "resources")
    POSTPROCESS_DIR = os.path.join(tappas_workspace, "apps/h8/gstreamer/libs/post_processes")
    CROPING_ALGORITHMS_DIR = os.path.join(POSTPROCESS_DIR, "cropping_algorithms")
    STREAM_ID_SO = os.path.join(POSTPROCESS_DIR, "libstream_id_tool.so")
    ADD_STREAM_ID_PATH = os.path.join(current_path, "add_stream_id.py")
    hailopython_path = os.path.join(current_path, "clip_hailopython.py")
    
    if (detector_pipeline == "fast_sam"):    
        # FASTSAM
        # DETECTION_HEF_PATH = os.path.join(RESOURCES_DIR, "fast_sam_s.hef")
        DETECTION_HEF_PATH = os.path.join(RESOURCES_DIR, "yolov8s_fastsam_single_context.hef")
        DETECTION_POST = os.path.join(RESOURCES_DIR, "libfastsam_post.so")
        detection_postprocess_so = DETECTION_POST
        DETECTION_POST_PIPE = f'hailofilter so-path={DETECTION_POST} qos=false '
        hef_path = DETECTION_HEF_PATH
    else:
        # personface
        YOLO5_POSTPROCESS_SO = os.path.join(POSTPROCESS_DIR, "libyolo_post.so")
        YOLO5_NETWORK_NAME = "yolov5_personface_letterbox"
        YOLO5_HEF_PATH = os.path.join(RESOURCES_DIR, "yolov5s_personface_reid.hef")
        YOLO5_CONFIG_PATH = os.path.join(RESOURCES_DIR, "configs/yolov5_personface.json")
        DETECTION_POST_PIPE = f'hailofilter so-path={YOLO5_POSTPROCESS_SO} qos=false function_name={YOLO5_NETWORK_NAME} config-path={YOLO5_CONFIG_PATH} '
        hef_path = YOLO5_HEF_PATH

    # CLIP 
    clip_hef_path = os.path.join(RESOURCES_DIR, "clip_resnet_50x4_adaround.hef")
    # clip_postprocess_so = os.path.join(POSTPROCESS_DIR, "libclip_post.so")
    clip_postprocess_so = os.path.join(RESOURCES_DIR, "libclip_post.so")
    DEFAULT_CROP_SO = os.path.join(CROPING_ALGORITHMS_DIR, "libvms_croppers.so")

    DEFAULT_VDEVICE_KEY = "1"
    
    # Initialize max_buffers_size as a placeholder
    max_buffers_size = 3
    QUEUE = f'queue leaky=no max-size-buffers={max_buffers_size} max-size-bytes=0 max-size-time=0 '
    BYPASS_QUEUE = f'queue leaky=no max-size-bytes=0 max-size-time=0 '
    # RATE_PIPELINE = f' videorate ! video/x-raw, framerate=30/1 ! {QUEUE} name=rate_queue '
    # Check if the input seems like a v4l2 device path (e.g., /dev/video0)
    # if re.match(r'/dev/video\d+', input_uri):
    #     SOURCE_PIPELINE = f'v4l2src device={input_uri} ! image/jpeg ! decodebin ! {RATE_PIPELINE} ! videoflip video-direction=horiz '
    # else:
    #     if re.match(r'0x\w+', input_uri): # Window ID - get from xwininfo
    #         SOURCE_PIPELINE = pipeline_str = f"ximagesrc xid={input_uri} ! {RATE_PIPELINE} ! videoconvert qos=false "
    #     else:
    #         SOURCE_PIPELINE = pipeline_str = f"uridecodebin uri={input_uri} ! {RATE_PIPELINE} ! videoconvert qos=false "
    REID_VIDEOS_DIR = os.path.join(tappas_workspace, "apps/h8/gstreamer/general/multi_person_multi_camera_tracking/resources/")    
    # define multi sources pipeline
    video_sources = [os.path.join(REID_VIDEOS_DIR,'reid0.mp4'),]
                    #  os.path.join(REID_VIDEOS_DIR,'reid1.mp4'),
                    #  os.path.join(REID_VIDEOS_DIR,'reid2.mp4'),
                    #  os.path.join(REID_VIDEOS_DIR,'reid3.mp4')]
    
    SOURCE_PIPELINE = 'hailoroundrobin name=rr_arbiter mode=2 '
    # SOURCE_PIPELINE = 'funnel name=rr_arbiter '
    for i in range(len(video_sources)):
        SOURCE_PIPELINE += f'uridecodebin uri=file://{video_sources[i]} name=uridecodebon_{i} ! '
        SOURCE_PIPELINE += f'{QUEUE} name=src_q{i} ! videoconvert name=convert_src_{i} qos=false ! '
        # SOURCE_PIPELINE += f'hailofilter name=set_id_{i} so-path={STREAM_ID_SO} config-path=SRC_{i} qos=false ! '
        # SOURCE_PIPELINE += f'hailopython name=set_id_{i} module={ADD_STREAM_ID_PATH} function=src_{i} qos=false ! '
        SOURCE_PIPELINE += f'rr_arbiter.sink_{i} '
    SOURCE_PIPELINE += ' rr_arbiter. '
    # SOURCE_PIPELINE = f'uridecodebin uri=file://{video_sources[0]} ! {RATE_PIPELINE} ! videoconvert qos=false '
    
    DETECTION_PIPELINE = f'{QUEUE} name=pre_detection_scale ! videoscale n-threads=4 qos=false ! \
        {QUEUE} name=pre_detecion_net ! \
        hailonet hef-path={hef_path} vdevice-key={DEFAULT_VDEVICE_KEY} batch-size={batch_size} scheduler-timeout-ms=100 ! \
        {QUEUE} name=pre_detecion_post ! \
        {DETECTION_POST_PIPE} ! \
        {QUEUE}'
    
    CLIP_PIPELINE = f'{QUEUE} name=pre_clip_net ! \
        hailonet hef-path={clip_hef_path} scheduling-algorithm=1 vdevice-key={DEFAULT_VDEVICE_KEY} batch-size={batch_size} ! \
        {QUEUE} ! \
        hailofilter so-path={clip_postprocess_so} qos=false ! \
        {QUEUE}'

    TRACKER = f'hailotracker name=hailo_face_tracker class-id=1 kalman-dist-thr=0.7 iou-thr=0.8 init-iou-thr=0.9 \
                keep-new-frames=2 keep-tracked-frames=2 keep-lost-frames=2 keep-past-metadata=true qos=false ! \
                {QUEUE} '
    
    DETECTION_PIPELINE_MUXER = f'{BYPASS_QUEUE} max-size-buffers=12 name=pre_detection_tee ! tee name=t hailomuxer name=hmux \
        t. ! {BYPASS_QUEUE} max-size-buffers=20 name=detection_bypass_q ! hmux.sink_0 \
        t. ! {DETECTION_PIPELINE} ! hmux.sink_1 \
        hmux. ! {QUEUE} '

    if detector_pipeline == "none":
        DETECTION_PIPELINE_WRAPPER = ""
    else:
        DETECTION_PIPELINE_WRAPPER = DETECTION_PIPELINE_MUXER

    # Clip pipeline with cropper integration
    CLIP_CROPPER_PIPELINE = f'hailocropper so-path={DEFAULT_CROP_SO} function-name=person_attributes internal-offset=true name=cropper \
        hailoaggregator name=agg \
        cropper. ! {BYPASS_QUEUE} max-size-buffers=20 name=clip_bypass_q ! agg.sink_0 \
        cropper. ! {CLIP_PIPELINE} ! agg.sink_1 \
        agg. ! {QUEUE} '
    
    # Clip pipeline with muxer integration (no cropper)
    CLIP_MUXER_PIPELINE = f'tee name=clip_t hailomuxer name=clip_hmux \
        clip_t. ! {BYPASS_QUEUE} max-size-buffers=20 name=clip_bypass_q ! clip_hmux.sink_0 \
        clip_t. ! videoscale n-threads=4 qos=false ! {CLIP_PIPELINE} ! clip_hmux.sink_1 \
        clip_hmux. ! {QUEUE} '

    # Display pipelines
    CLIP_DISPLAY_PIPELINE = f'{QUEUE} ! \
                            fpsdisplaysink name=hailo_display sync={sync} text-overlay=true '
    
    def CLIP_SMALL_DISPLAY_PIPELINE(i):
        return f'{QUEUE} ! videoscale ! video/x-raw, widht=480, height=270 ! \
                            fpsdisplaysink name=hailo_display_{i} sync={sync} text-overlay=true '
    
    STREAM_ROUTER_PIPELINE = f'hailostreamrouter name=sid '
    STREAM_ROUTER_PIPELINE_END = ''
    for i in range(len(video_sources)):
        STREAM_ROUTER_PIPELINE += f'src_{i}::input-streams="<sink_{i}>" '
        STREAM_ROUTER_PIPELINE_END += f'sid.src_{i} ! {CLIP_SMALL_DISPLAY_PIPELINE(i)} '
    STREAM_ROUTER_PIPELINE += STREAM_ROUTER_PIPELINE_END

    # Text to image matcher
    CLIP_POSTPROCESS_PIPELINE = f' hailopython name=pyproc module={hailopython_path} qos=false ! \
        {QUEUE} ! \
        hailooverlay local-gallery=false show-confidence=true font-thickness=2 qos=false ! \
        {QUEUE} ! videoconvert n-threads=4 qos=false ! {QUEUE} '
    
    # PIPELINE
    if detector_pipeline == "none":
        PIPELINE = f'{SOURCE_PIPELINE} ! \
            {CLIP_MUXER_PIPELINE} ! \
            {CLIP_POSTPROCESS_PIPELINE} ! \
            {STREAM_ROUTER_PIPELINE}'
    else:
        # PIPELINE = f'{SOURCE_PIPELINE} ! \
        #     {DETECTION_PIPELINE_WRAPPER} ! \
        #     {TRACKER} ! \
        #     {CLIP_CROPPER_PIPELINE} ! \
        #     {CLIP_POSTPROCESS_PIPELINE} ! \
        #     {CLIP_DISPLAY_PIPELINE}'
        #     # {STREAM_ROUTER_PIPELINE}'
        PIPELINE = f'{SOURCE_PIPELINE} ! \
            {DETECTION_PIPELINE_WRAPPER} ! \
            {TRACKER} ! \
            {CLIP_CROPPER_PIPELINE} ! \
            {CLIP_POSTPROCESS_PIPELINE} ! \
            {STREAM_ROUTER_PIPELINE}'
            # {CLIP_DISPLAY_PIPELINE}'

    return PIPELINE
