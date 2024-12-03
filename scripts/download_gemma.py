import modal

MODELS_DIR = "/gemma"

DEFAULT_NAME = "google/gemma-2-9b"
DEFAULT_REVISION = "main"

volume = modal.Volume.from_name("gemma", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        [
            "huggingface_hub==0.26.3",  # download models from the Hugging Face Hub
            "hf-transfer==0.1.9",  # download models faster with Rust
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App(
    image=image,
    secrets=[  # add a Hugging Face Secret if you need to download a gated model
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
    ],
)


@app.function(volumes={MODELS_DIR: volume}, timeout=4 * HOURS)
def download_model(model_name, model_revision, force_download=False):
    from huggingface_hub import snapshot_download

    print(f"Starting download for {model_name} at revision {model_revision}")

    volume.reload()

    snapshot_download(
        model_name,
        local_dir=MODELS_DIR + "/" + model_name,
        ignore_patterns=[
            "*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ],  # Ensure safetensors
        revision=model_revision,
        force_download=force_download,
    )

    volume.commit()


@app.local_entrypoint()
def main(
    model_name: str = DEFAULT_NAME,
    model_revision: str = DEFAULT_REVISION,
    force_download: bool = False,
):
    download_model.remote(model_name, model_revision, force_download)
