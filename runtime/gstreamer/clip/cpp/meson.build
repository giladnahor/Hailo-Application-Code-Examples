
################################################
# CLIP SOURCES
################################################
clip_sources = [
    'clip.cpp',
]
shared_library('clip_post',
    clip_sources,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: join_paths(meson.project_source_root(), 'resources'),
)    
################################################
# fastsam SOURCES
################################################
fastsam_sources = [
    'fastsam_postprocess.cpp',
]
shared_library('fastsam_post',
    fastsam_sources,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: join_paths(meson.project_source_root(), 'resources'),
)    
