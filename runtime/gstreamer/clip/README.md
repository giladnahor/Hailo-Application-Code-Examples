# Clip Inference Example

This is an example application to run a CLIP inference on a video in realtime.
The image embeddings are accelerated by Hailo-8 AI processor.
The text embeddings are running on the host. Text embeddings are sparse and should be calculated only once per text. If they do not need to be updated in real time they can be saved to a JSON file and loaded on the next run.
As default the app starts w/o enabling online test embeddings. This is done to speed up load time and save memory. It also allows to run on low memory hosts like the RPi.

## Prerequisites
This example was tested on Hailo Suite 2023_10 (or TAPPAS 3.26.0)
You'll need to have them installed on your system.
### hailo_tappas pkg_config
This applications is using hailo_tapps pkg_config. 
If this command returns TAPPAS version you are good to go:
```bash
    pkg-config --modversion hailo_tappas
```
If not make sure:
- You have read rights to /opt/hailo/tappas/pkgconfig/hailo_tappas.pc
- PKG_CONFIG_PATH includes this path.
To add it to PKG_CONFIG_PATH run:
```bash
    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/hailo/tappas/pkgconfig/
```

## CPP code compilation
Some CPP code is used in this app for post processes and cropping. These should be compiled before running the example. It uses Hailo pkg-config to find the required libraries.
Some requirements are missing in the current release, so we need to add them manually. (Hailo Suite 2023_10)

!!! info You might need sudo permissions.

```bash
    cp /opt/hailo/tappas/pkgconfig/hailo_tappas.pc /opt/hailo/tappas/pkgconfig/hailo_tappas.pc.bkp
    cp cpp/hailo_tappas.pc /opt/hailo/tappas/pkgconfig/
```

!!! warning After modifying the hailo_tappas.pc run diff between the original version and the modified one. If you are using aarm64 and not x86_64 make sure to edit arch field. arch=x86_64 should be changed to arch=aarch64.The TAPPAS_WORKSPACE might also need to be changed to the correct path. For these 2 fields use the original hailo_tappas.pc.bkp as reference.

The compilation script is **compile_postprocess.sh** you can run it manually but it will be ran automatically when installing the package.
The postprocess so files will be installed under resources dir:
- libclip_post.so
- libfastsam_post.so
- libclip_croppers.so


## installation
!!! warning Make sure you have Hailo TAPPAS venv activated.
To install the application run in the application root dir:
```bash 
    python3 -m pip install -e .
```
This will install the app as a python package in "Editable" mode. It will also compile the CPP code and download required HEF files.

## Usage
!!! info Hailo venv should be activated. 

Run the example:
```bash
clip_app
```
It can also be ran directly from the source (from the application root dir):
```bash
python3 -m clip_app.clip_app
```

On the first run clip will download the required models. This will happen only once.

!!! info TAPPAS_WORKSPACE and TAPPAS_VERSION are read from the pkg_config. You can set them manually by setting them as enviromnet variables.

### Arguments
```bash
clip_app -h
usage: clip_app [-h] [--sync] [--input INPUT] [--dump-dot] [--detection-threshold DETECTION_THRESHOLD] [--detector {person,none}] [--onnx-runtime] [--clip-runtime]
                [--json-path JSON_PATH] [--multi-stream]

Hailo online clip app

options:
  -h, --help            show this help message and exit
  --sync                Enable display sink sync.
  --input INPUT, -i INPUT
                        URI of the input stream.
  --dump-dot            Dump the pipeline graph to a dot file.
  --detection-threshold DETECTION_THRESHOLD
                        Detection threshold
  --detector {person,none}, -d {person,none}
                        Which detection pipeline to use.
  --onnx-runtime        Not supported. Requires ONNX runtime for text embedding.
  --clip-runtime        When set app will use clip pythoch runtime for text embedding.
  --json-path JSON_PATH
                        Path to json file to load and save embeddings. If not set embeddings.json will be used.
  --multi-stream        When set app will use multi stream pipeline. In this mode detector is set to person.
```

### Modes
- The default mode (--detector none) will run only the clip inference on the entire frame. This is the mode CLIP is trained for, and will give the best results. In this mode CLIP is acting as a classifier describing the entire frame. CLIP will be ran on every frame.
- Person mode (--detector person) will run the clip inference only on the detected persons. In this mode we first run a person detector and then run clip on the detected persons. This mode does not exaclty fit the data base CLIP was trained on but gives a good use case for using CLIP as a detector. In this mode CLIP is acting as a person classifier. In this mode CLIP will be ran only on detected persons. To reduce the number of clip inferences we run clip only every second per tracked person. This can be changed in the code.
- Multi stream mode (--multi-stream) will run clip on 4 streams. This mode is presenting a use case of using CLIP to monitor multiple streams. In this mode we are running with a person detector. The App will bring the most relevant stream to the foreground. If the searched person is not found in any of the streams a random stream will be selected.
  
!!! info Multi stream mode requires a strong machine. 

### Online text embeddings
- To run with online text embeddings run with the --clip-runtime flag. This will run the text embeddings on the host. You will be able to change the text on the fly. This mode might not work on weak machines. It requires a host with enough memory to run the text embeddings model (run on CPU).
- You can set which json file to use for saving and loading embeddings using the --json-path flag. If not set embeddings.json will be used.


!!! info As default the online text embeddings are disabled. This is done to speed up load time and save memory. It also allows to run on low memory hosts like the RPi. 

### Offline text embeddings
- You can save the embeddings to a json file and load them on the next run. This will not require to run the text embeddings on the host.
- If you need to prepare text embeddings on a weak machine you can use the 'text_image_matcher' tool. This tool will run the text embeddings on the host and save them to a json file. Without running the intire pipeline.
``` bash
text_image_matcher -h
usage: text_image_matcher [-h] [--output OUTPUT] [--interactive] [--image-path IMAGE_PATH]

options:
  -h, --help            show this help message and exit
  --output OUTPUT       output file name default=text_embeddings.json
  --interactive         input text from interactive shell
  --image-path IMAGE_PATH
                        Optional, path to image file to match. Note image embeddings are not running on Hailo here.
```
### Known issues
On a low memory host HailoRT driver might no be able to allocate the required memory. 
The error looks like this:
```bash
[HailoRT] [error] Failed to allocate buffer with errno: 12
[HailoRT] [error] CHECK_EXPECTED failed with status=HAILO_DRIVER_FAIL(36)
[HailoRT] [error] CHECK_EXPECTED failed with status=HAILO_DRIVER_FAIL(36)
......
```
This is not a Driver issue but a memory issue. The error repoert will be updated in the next release to reflect this.

To resolve this issue you can:
- If you believe you have enough memory you can just rerun and try to allocate the memory again. (It might be a defragmentation issue)
- You can try to free some memory by killing some processes.
- If you are running with online text embeddings you can try to run with offline text embeddings. This will save some memory.
- You can lower the batch size. This might reduce performance. (in clip_pipeline.py change the batch size to 1)