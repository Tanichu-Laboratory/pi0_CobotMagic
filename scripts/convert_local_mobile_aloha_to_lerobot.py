"""
Pack the local mobile Aloha demonstrations into a LeRobot dataset."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import Literal, Sequence

import h5py
import numpy as np
import torch
import tqdm
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

# dict{data_dir_name : task prompt}
DEFAULT_PROMPTS = {
    "pen": "uncap the pen and leave it on the table",
    "pet": "uncap the plastic bottle",
    "ziploc": "open the ziploc bag",
    "ziploc2": "open the ziploc bag",
}

CAMERA_ALIASES = {
    "cam_high": "top",
    "cam_low": "low",
    "cam_left_wrist": "left_wrist",
    "cam_right_wrist": "right_wrist",
}


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    cameras: Sequence[str],
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]
    features = {
        "state": {"dtype": "float32", "shape": (len(motors),), "names": [motors]},
        "observation.state": {"dtype": "float32", "shape": (len(motors),), "names": [motors]},
        "action": {"dtype": "float32", "shape": (len(motors),), "names": [motors]},
    }

    if has_velocity:
        features["observation.velocity"] = {"dtype": "float32", "shape": (len(motors),), "names": [motors]}
    if has_effort:
        features["observation.effort"] = {"dtype": "float32", "shape": (len(motors),), "names": [motors]}

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    dataset_path = HF_LEROBOT_HOME / repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def infer_camera_keys(hdf5_files: list[list[Path]]) -> list[tuple[str, str]]:
    shared_keys: set[str] | None = None
    for file_list in hdf5_files:
        if not file_list:
            continue
        with h5py.File(file_list[0], "r") as ep:
            keys = {key for key in ep["/observations/images"].keys() if "depth" not in key}
        shared_keys = keys if shared_keys is None else shared_keys & keys
    if not shared_keys:
        raise ValueError("No overlapping RGB camera keys were found across tasks.")
    camera_pairs: list[tuple[str, str]] = []
    for raw in sorted(shared_keys):
        camera_pairs.append((raw, CAMERA_ALIASES.get(raw, raw)))
    return camera_pairs


def load_raw_images_per_camera(ep: h5py.File, cameras: Sequence[tuple[str, str]]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for raw_name, canonical in cameras:
        dataset_key = f"/observations/images/{raw_name}"
        uncompressed = ep[dataset_key].ndim == 4
        if uncompressed:
            imgs_array = ep[dataset_key][:]
        else:
            import cv2

            imgs_array = []
            for data in ep[dataset_key]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)
        imgs_per_cam[canonical] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
    cameras: Sequence[tuple[str, str]],
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])
        velocity = torch.from_numpy(ep["/observations/qvel"][:]) if "/observations/qvel" in ep else None
        effort = torch.from_numpy(ep["/observations/effort"][:]) if "/observations/effort" in ep else None
        imgs_per_cam = load_raw_images_per_camera(ep, cameras)
    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    cameras: Sequence[tuple[str, str]],
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]
        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path, cameras)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "state": state[i],
                "action": action[i],
                "task": task,
            }
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]
            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


@dataclasses.dataclass
class ConversionArgs:
    data_root: Path = Path("project/data")
    repo_id: str = "local/mobile_aloha_multitask"
    dataset_subdir: str = "aloha_mobile_dummy"
    push_to_hub: bool = False
    robot_type: Literal["aloha", "mobile_aloha"] = "mobile_aloha"
    mode: Literal["video", "image"] = "image"


def discover_tasks(data_root: Path, dataset_subdir: str) -> dict[str, list[Path]]:
    tasks: dict[str, list[Path]] = {}
    for task_dir in sorted(data_root.iterdir()):
        if not task_dir.is_dir():
            continue
        episode_dir = task_dir / dataset_subdir
        if not episode_dir.is_dir():
            continue
        hdf5_files = sorted(episode_dir.glob("episode_*.hdf5"))
        if not hdf5_files:
            continue
        tasks[task_dir.name] = hdf5_files
    if not tasks:
        raise FileNotFoundError(f"No episode_* hdf5 files found under {data_root}")
    return tasks


def build_dataset_config(mode: Literal["video", "image"]) -> DatasetConfig:
    return DatasetConfig(
        use_videos=mode == "video",
        tolerance_s=DEFAULT_DATASET_CONFIG.tolerance_s,
        image_writer_processes=DEFAULT_DATASET_CONFIG.image_writer_processes,
        image_writer_threads=DEFAULT_DATASET_CONFIG.image_writer_threads,
        video_backend=DEFAULT_DATASET_CONFIG.video_backend,
    )


def main(args: ConversionArgs) -> None:
    if not args.data_root.exists():
        raise FileNotFoundError(f"{args.data_root} does not exist")

    tasks = discover_tasks(args.data_root, args.dataset_subdir)
    dataset_config = build_dataset_config(args.mode)

    task_lists = list(tasks.values())
    camera_pairs = infer_camera_keys(task_lists)
    has_vel = all(has_velocity(paths) for paths in task_lists)
    has_eff = all(has_effort(paths) for paths in task_lists)
    canonical_cameras = [canonical for _, canonical in camera_pairs]

    dataset = create_empty_dataset(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        cameras=canonical_cameras,
        mode=args.mode,
        has_velocity=has_vel,
        has_effort=has_eff,
        dataset_config=dataset_config,
    )

    for task_name, episode_paths in tasks.items():
        prompt = DEFAULT_PROMPTS.get(task_name, f"perform the {task_name} task")
        print(f"[convert] exporting {task_name} ({len(episode_paths)} episodes) as task='{prompt}'")
        populate_dataset(dataset, episode_paths, task=prompt, cameras=camera_pairs)

    # All episodes are saved individually; no explicit consolidation step needed.
    if args.push_to_hub:
        dataset.push_to_hub()

    output_dir = HF_LEROBOT_HOME / args.repo_id
    print(f"[convert] wrote dataset to {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(ConversionArgs))
