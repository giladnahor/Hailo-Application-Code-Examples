# Clip Inference Example

This is an example application to run a CLIP inference on a video in realtime.

## Prerequisites
This example was tested on Hailo Suite 2023_10 (or TAPPAS 3.26.0)
You'll need to have them installed on your system.
### hailo_tappas pkg_config
This applications is using hailo_tapps pkg_config. 
If this command returns TAPPAS version you are good to go:
```bash
    pkg-config --modversion hailo_tappas
```
If not make sure you have read rights to /opt/hailo/tappas/pkgconfig/hailo_tappas.pc and that PKG_CONFIG_PATH includes this path.
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
The compilation script is **compile_postprocess.sh** you can run it manually but it will be ran automatically when installing the package.
The postprocess so files will be installed under resources dir:
- libclip_post.so
- libfastsam_post.so


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
