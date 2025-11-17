from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="xlangai/OpenCUA-7B",
    local_dir="OpenCUA-7B",                
    local_dir_use_symlinks=False  
)