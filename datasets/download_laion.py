from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="laion/laions_got_talent_enhanced_flash_annotations_and_long_captions",
    repo_type="dataset",
    local_dir="./laion_talent",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=8,
)