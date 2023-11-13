# Clip Inference Example

This is an example application to run a CLIP inference on a video in realtime.
The image embeddings are accelerated by Hailo-8 AI processor.
The text embeddings are running on the host. Text embeddings are sparse and should be calculated only once per text. If they do not need to be updated in real time they can be saved to a JSON file and loaded on the next run.
As default the app starts w/o enabling online test embeddings. This is done to speed up load time and save memory. It also allows to run on low memory hosts like the RPi4.

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

## post process compilation
The clip postprocess should be compiled before running the example. It uses Hailo pkg-config to find the required libraries.
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
!!! warning Make sure you have the correct Hailo venv activated.
To install the application run in the application root dir:
```bash 
    python3 -m pip install -e .
```


## Usage
!!! info Hailo venv shuold be activated. 

Run the example:
```bash
python3 clip_app.py
```
It can also be run simply by:
```bash
clip_app
```

On the first run clip will download the required models. This will happen only once.

!!! info TAPPAS_WORKSPACE and TAPPAS_VERSION are read from the pkg_config. You can set them manually by setting them as enviromnet variables.

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