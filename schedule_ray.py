import ray
import os
import toml
import argparse
from ray import init, remote
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from train import train_func  # your existing train.py must expose this

# Set NCCL flags (equivalent to your shell command)
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

init(ignore_reinit_error=True, log_to_driver=True)

@remote
def launch_training(resume_from_checkpoint=None):
    # config_path = "examples/wan_video.toml"
    # config_path = os.path.abspath("examples/wan_video.toml")
    config_path = "/app/AdFame/trainig_pipeline/diffusion-pipe/examples/wan_video.toml"
    # config_path = "/home/cc/bhumi/AdFame/trainig_pipeline/diffusion-pipe/examples/wan_video.toml"


    toml_config = toml.load(config_path)

    args = argparse.Namespace(
        config=config_path,
        local_rank=0,
        resume_from_checkpoint=resume_from_checkpoint,
        regenerate_cache=None,
        cache_only=False,
        i_know_what_i_am_doing=False
    )

    ray_config = {
        "args": args,
        "toml_config": toml_config
    }

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=ray_config,
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=True,
            resources_per_worker={"CPU": 32, "GPU": 1}
        ),
    )

    return trainer.fit()

def find_latest_checkpoint(output_dir):
    folders = sorted(os.listdir(output_dir), reverse=True)
    for f in folders:
        path = os.path.join(output_dir, f)
        if os.path.isdir(path) and any(fname.startswith("step") or "ckpt" in fname for fname in os.listdir(path)):
            return f
    return None

if __name__ == "__main__":
    choice = input("▶️ Do you want to train or retrain? [train/retrain]: ").strip().lower()
    config_path = "/app/AdFame/trainig_pipeline/diffusion-pipe/examples/wan_video.toml"
    output_dir = toml.load(config_path)["output_dir"]

    if choice == "train":
        ray.get(launch_training.remote(resume_from_checkpoint=None))

    elif choice == "retrain":
        latest_ckpt = find_latest_checkpoint(output_dir)
        if latest_ckpt:
            print(f"📦 Found checkpoint: {latest_ckpt}")
            ray.get(launch_training.remote(resume_from_checkpoint=latest_ckpt))

        else:
            print("❌ No valid checkpoint found in output directory.")

    else:
        print("⚠️ Invalid choice. Please enter 'train' or 'retrain'.")
