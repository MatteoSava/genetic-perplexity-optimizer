import modal
from pathlib import Path

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

app_name = "santa-2024"
app = modal.App(app_name)
gpu = "H100"
volume_name = "gemma"

try:
    volume = modal.Volume.lookup(volume_name, create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_gemma.py")


base_image = modal.Image.debian_slim(python_version="3.11").pip_install_from_pyproject(
    "pyproject.toml"
)

torch_image = base_image.pip_install(
    "torch==2.1.2",
    "tensorboard==2.17.1",
    "numpy==1.26.4",
)

volume_path = Path("/modal")
# model_filename = "nano_gpt_model.pt"
# best_model_filename = "best_nano_gpt_model.pt"
# tb_log_path = volume_path / "tb_logs"
# model_save_path = volume_path / "models"


mounts = [
    modal.Mount.from_local_dir(
        Path(__file__).parent.parent / "data", remote_path=Path("/root/data")
    ),
    modal.Mount.from_local_dir(
        Path(__file__).parent.parent / "genetic", remote_path=Path("/root/genetic")
    ),
]


@app.function(
    image=torch_image,
    secrets=[  # add a Hugging Face Secret if you need to download a gated model
        modal.Secret.from_name("wandb-secret")
    ],
    mounts=mounts,
    volumes={volume_path: volume},
    gpu=gpu,
    timeout=1 * HOURS,
)
def main():
    from genetic import main

    main.train()
