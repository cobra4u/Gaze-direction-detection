# FILE: ~/gazemain/loader.py
# ETH-XGaze (xgaze_224) HDF5 DataLoader with demographic label integration.
# Now reads an optional demographics CSV to add race, gender, and age labels to each sample.

import os
import glob
import math
import random
import warnings
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Avoid locking issues on shared FS
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

try:
    import h5py
except Exception as e:
        raise ImportError("Please 'pip install h5py' in your environment.") from e

try:
    import torchvision.transforms as T
except Exception as e:
    raise ImportError("Please 'pip install torchvision' in your environment.") from e


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- NEW: Mappings for demographic labels ---
# These convert string labels from the CSV into integer indices for the model.
RACE_TO_IDX = {
    'asian': 0, 'indian': 1, 'black': 2, 'white': 3,
    'middle eastern': 4, 'latino hispanic': 5, 'unknown': 6
}
GENDER_TO_IDX = {'Woman': 0, 'Man': 1, 'unknown': 2}
AGE_BIN_TO_IDX = {'0-17': 0, '18-34': 1, '35-54': 2, '55+': 3, 'unknown': 4}


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np
        np.random.seed(worker_seed)
    except Exception:
        pass


def build_transforms(face_size: int = 224, augment: bool = True, color_jitter: bool = True):
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if augment:
        ops = [T.Resize((face_size, face_size))]
        if color_jitter:
            ops.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02))
        ops += [T.ToTensor(), normalize]
        return T.Compose(ops)
    else:
        return T.Compose([T.Resize((face_size, face_size)), T.ToTensor(), normalize])


def vector_to_angles(vx: float, vy: float, vz: float) -> Tuple[float, float]:
    vy = float(np.clip(vy, -1.0, 1.0))
    pitch = math.asin(vy)
    yaw = math.atan2(vx, -float(np.clip(vz, -1.0, 1.0)))
    return yaw, pitch


class XGazeH5Dataset(Dataset):
    """
    HDF5 dataset for ETH-XGaze, now with demographic label support.
    
    If a `demographics_csv_path` is provided, it loads the CSV and attaches
    race, gender, and age_bin labels to each sample.

    sample = {
        'face': Tensor[3,H,W],
        'gaze': Tensor[2] (yaw, pitch) radians,
        'head_pose': Tensor[2] (optional),
        'subject_id': str,
        # --- NEW: Optional demographic labels ---
        'race': Tensor[] (long),
        'gender': Tensor[] (long),
        'age_bin': Tensor[] (long),
    }
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        face_size: int = 224,
        training: bool = False,
        hflip_prob: float = 0.0,
        color_jitter: bool = True,
        # --- NEW: Path to our demographics file ---
        demographics_csv_path: Optional[str] = None,
        # ... (rest of the arguments are the same)
        assume_angles_in_degrees: bool = False,
        include_eyes: bool = False,
        bgr_to_rgb: bool = False,
        strict: bool = True,
        include_subjects: Optional[Set[str]] = None,
        exclude_subjects: Optional[Set[str]] = None,
        gaze_key_hint: Optional[str] = None,
        allow_head_pose_as_gaze: bool = False,
        return_head_pose: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.training = training
        self.hflip_prob = hflip_prob
        self.assume_angles_in_degrees = assume_angles_in_degrees
        self.include_eyes = include_eyes
        self.bgr_to_rgb = bgr_to_rgb
        self.include_subjects = include_subjects
        self.exclude_subjects = exclude_subjects
        self.gaze_key_hint = gaze_key_hint
        self.allow_head_pose_as_gaze = allow_head_pose_as_gaze
        self.return_head_pose = return_head_pose

        self.face_tf = build_transforms(face_size=face_size, augment=training, color_jitter=color_jitter)

        # --- NEW: Load and process demographics CSV ---
        self.demographics_lookup = None
        if demographics_csv_path:
            if not os.path.exists(demographics_csv_path):
                warnings.warn(f"Demographics CSV not found at: {demographics_csv_path}. Labels will not be loaded.")
            else:
                try:
                    df = pd.read_csv(demographics_csv_path)
                    # Convert dataframe to a dictionary for fast lookup: {subject_id: {col: val, ...}}
                    self.demographics_lookup = df.set_index('subject_id').to_dict('index')
                    print(f"Loaded {len(self.demographics_lookup)} demographic records from {demographics_csv_path}")
                except Exception as e:
                    warnings.warn(f"Failed to load or parse demographics CSV: {e}")
        # -------------------------------------------

        # Collect H5 files (this part is unchanged)
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.h5")))
        if not self.files:
            self.files = sorted(glob.glob(os.path.join(split_dir, "**", "*.h5"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in {split_dir}")

        # Subject filters (unchanged)
        def subj_from_path(p: str) -> str:
            return os.path.splitext(os.path.basename(p))[0]

        if self.include_subjects is not None:
            self.files = [f for f in self.files if subj_from_path(f) in self.include_subjects]
        if self.exclude_subjects is not None:
            self.files = [f for f in self.files if subj_from_path(f) not in self.exclude_subjects]
        if not self.files:
            raise FileNotFoundError(f"No .h5 files left after subject filtering in {split_dir}")

        # Key detection and indexing (unchanged)
        self.img_key = None
        self.gaze_key = None
        self.head_pose_key = None
        self.index_map: List[Tuple[int, int]] = []
        self._h5_cache: Dict[int, h5py.File] = {}

        self._detect_keys_from_example(self.files[0])
        for fi, fpath in enumerate(self.files):
            try:
                with h5py.File(fpath, "r") as f:
                    N = self._dataset_length(f, self.img_key)
                if N <= 0:
                    if strict:
                        raise ValueError(f"Empty dataset '{self.img_key}' in file: {fpath}")
                    else:
                        warnings.warn(f"Skipping empty file: {fpath}")
                        continue
                self.index_map.extend([(fi, i) for i in range(N)])
            except Exception as e:
                warnings.warn(f"Could not read H5 file {fpath}, skipping. Error: {e}")


        if not self.index_map:
            raise RuntimeError("No samples indexed. Check HDF5 keys/content.")

    # --- HDF5 helpers and key detection methods are unchanged ---
    # (Methods like _open_h5, _close_all, __del__, _list_datasets, _detect_keys_from_example,
    # _dataset_length, _load_image, _load_gaze, _maybe_hflip are identical to before)
    
    def _open_h5(self, file_idx: int) -> h5py.File:
        h5f = self._h5_cache.get(file_idx)
        if h5f is None:
            h5f = h5py.File(self.files[file_idx], "r", libver="latest", swmr=True)
            self._h5_cache[file_idx] = h5f
        return h5f

    def _close_all(self):
        for k, f in list(self._h5_cache.items()):
            try:
                f.close()
            except Exception:
                pass
            self._h5_cache.pop(k, None)

    def __del__(self):
        try:
            if hasattr(self, "_h5_cache"):
                self._close_all()
        except Exception:
            pass

    def _list_datasets(self, group, prefix="") -> List[Tuple[str, h5py.Dataset]]:
        items = []
        for k, v in group.items():
            path = f"{prefix}{k}"
            if isinstance(v, h5py.Dataset):
                items.append((path, v))
            elif isinstance(v, h5py.Group):
                items.extend(self._list_datasets(v, prefix=path + "/"))
        return items

    def _detect_keys_from_example(self, example_path: str):
        with h5py.File(example_path, "r") as f:
            datasets = self._list_datasets(f)
            candidates_img = [
                (p, ds.shape) for p, ds in datasets
                if ds.ndim == 4 and (ds.shape[-1] == 3 or ds.shape[1] == 3)
            ]
            if not candidates_img:
                raise KeyError(f"No image dataset found in {example_path}")
            
            def img_score(name: str):
                n = name.lower().split("/")[-1]
                score = 1 if "face" in n else 0
                if "eye" in n: score -= 2
                return score
            candidates_img.sort(key=lambda x: (img_score(x[0]), -x[1][0]), reverse=True)
            self.img_key = candidates_img[0][0]

            self.head_pose_key = next((p for p, ds in datasets if "head" in p.lower() and "pose" in p.lower() and ds.ndim == 2 and ds.shape[1] in (2, 3)), None)
            
            candidates_gaze = [
                (p, ds.shape[1]) for p, ds in datasets
                if "gaze" in p.lower() and "head" not in p.lower() and ds.ndim == 2 and ds.shape[1] in (2, 3)
            ]

            if self.gaze_key_hint and self.gaze_key_hint in dict(datasets):
                 self.gaze_key = self.gaze_key_hint
            elif candidates_gaze:
                candidates_gaze.sort(key=lambda x: x[1], reverse=True)
                self.gaze_key = candidates_gaze[0][0]
            elif self.allow_head_pose_as_gaze and self.head_pose_key:
                warnings.warn(f"Falling back to head_pose as gaze for {example_path}.")
                self.gaze_key = self.head_pose_key
            else:
                raise KeyError(f"No proper gaze dataset found in {example_path}")

    def _dataset_length(self, f: h5py.File, key: str) -> int:
        return f[key].shape[0]

    def _load_image(self, f: h5py.File, key: str, idx: int) -> Image.Image:
        arr = np.asarray(f[key][idx])
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        if self.bgr_to_rgb:
            arr = arr[..., ::-1]
        return Image.fromarray(arr.astype(np.uint8), "RGB")

    def _load_gaze(self, f: h5py.File, idx: int) -> Tuple[float, float]:
        g = np.asarray(f[self.gaze_key][idx]).astype(np.float32)
        if g.shape[-1] == 3:
            return vector_to_angles(g[0], g[1], g[2])
        yaw, pitch = float(g[0]), float(g[1])
        if self.assume_angles_in_degrees:
            yaw, pitch = np.deg2rad([yaw, pitch])
        return yaw, pitch

    def _maybe_hflip(self, face_img: Image.Image) -> Tuple[bool, Image.Image]:
        do_flip = self.training and random.random() < self.hflip_prob
        if do_flip:
            face_img = face_img.transpose(Image.FLIP_LEFT_RIGHT)
        return do_flip, face_img

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, global_idx: int) -> Dict:
        file_idx, idx_within = self.index_map[global_idx]
        h5f = self._open_h5(file_idx)

        face_img = self._load_image(h5f, self.img_key, idx_within)
        did_flip, face_img = self._maybe_hflip(face_img)
        face_t = self.face_tf(face_img)

        yaw, pitch = self._load_gaze(h5f, idx_within)
        if did_flip:
            yaw = -yaw

        fpath = self.files[file_idx]
        subject_id = os.path.splitext(os.path.basename(fpath))[0]

        sample = {
            "face": face_t,
            "gaze": torch.tensor([yaw, pitch], dtype=torch.float32),
            "subject_id": subject_id,
        }

        if self.return_head_pose and self.head_pose_key is not None:
            hp = np.asarray(h5f[self.head_pose_key][idx_within]).astype(np.float32)
            if hp.shape[-1] == 2:
                h_yaw, h_pitch = float(hp[0]), float(hp[1])
            elif hp.shape[-1] == 3:
                h_yaw, h_pitch = vector_to_angles(hp[0], hp[1], hp[2])
            else:
                h_yaw, h_pitch = 0.0, 0.0
            if did_flip:
                h_yaw = -h_yaw
            sample["head_pose"] = torch.tensor([h_yaw, h_pitch], dtype=torch.float32)
            
        # --- NEW: Add demographic labels if lookup is available ---
        if self.demographics_lookup:
            demog_data = self.demographics_lookup.get(subject_id)
            if demog_data:
                sample['race'] = torch.tensor(RACE_TO_IDX.get(demog_data['race'], RACE_TO_IDX['unknown']), dtype=torch.long)
                sample['gender'] = torch.tensor(GENDER_TO_IDX.get(demog_data['gender'], GENDER_TO_IDX['unknown']), dtype=torch.long)
                sample['age_bin'] = torch.tensor(AGE_BIN_TO_IDX.get(demog_data['age_bin'], AGE_BIN_TO_IDX['unknown']), dtype=torch.long)
        # -----------------------------------------------------------

        return sample

# --- The factory functions below are now updated to pass the demographic CSV path ---

def _build_common_loader_kwargs(num_workers: int):
    return dict(
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None, worker_init_fn=seed_worker,
    )

def _get_demographic_paths(base_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    FIXED: Intelligently finds train and test CSV paths from a base path.
    Example: base_path = '.../metadata/xgaze_train_demographics.csv'
    Returns: ('.../xgaze_train_demographics.csv', '.../xgaze_test_demographics.csv')
    """
    if not base_path:
        return None, None

    if "_train" in base_path:
        train_path = base_path
        test_path = base_path.replace("_train", "_test")
    elif "_test" in base_path:
        test_path = base_path
        train_path = base_path.replace("_test", "_train")
    else:
        # Fallback if the name doesn't follow the convention
        train_path, test_path = base_path, base_path

    if not os.path.exists(test_path):
        warnings.warn(f"Could not find corresponding test demographics file at {test_path}")
        test_path = None # Set to None if it doesn't exist

    return train_path, test_path


def create_xgaze_h5_dataloaders(
    root_dir: str,
    batch_size_train: int = 128,
    batch_size_eval: int = 256,
    num_workers: int = 8,
    demographics_csv_path: Optional[str] = None, # NEW
    **kwargs
):
    # drop unsupported keys from kwargs
    kwargs.pop("distributed", None)
    """Factory for train/test loaders without a separate validation set."""
    # FIXED LOGIC
    train_demog_path, test_demog_path = _get_demographic_paths(demographics_csv_path)

    train_set = XGazeH5Dataset(root_dir=root_dir, split="train", training=True, demographics_csv_path=train_demog_path, **kwargs)
    
    test_set = None
    try:
        test_set = XGazeH5Dataset(root_dir=root_dir, split="test", training=False, demographics_csv_path=test_demog_path, **kwargs)
    except (KeyError, FileNotFoundError):
        warnings.warn("Could not create a 'test' loader. Trying 'test_person_specific'.")
        try:
             test_set = XGazeH5Dataset(root_dir=root_dir, split="test_person_specific", training=False, demographics_csv_path=test_demog_path, strict=False, **kwargs)
        except (KeyError, FileNotFoundError):
             warnings.warn("No valid test set found.")

    common = _build_common_loader_kwargs(num_workers)
    train_loader = DataLoader(train_set, batch_size=train_size, shuffle=True, drop_last=True, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size_eval, shuffle=False, drop_last=False, **common) if test_set else None

    return {"train": train_loader, "test": test_loader}


def create_xgaze_h5_dataloaders_with_val(
    root_dir: str,
    val_subjects_file: str,
    batch_size_train: int = 128,
    batch_size_eval: int = 256,
    num_workers: int = 8,
    demographics_csv_path: Optional[str] = None, # NEW
    **kwargs
):
    # drop unsupported keys from kwargs
    kwargs.pop("distributed", None)
    # drop unsupported keys from kwargs
    kwargs.pop("distributed", None)
    """Factory for train/val/test loaders with subject-wise split."""
    with open(val_subjects_file, "r") as f:
        val_subjects = {ln.strip() for ln in f if ln.strip()}

    # FIXED LOGIC
    train_demog_path, test_demog_path = _get_demographic_paths(demographics_csv_path)

    # Train and Val sets both use the training demographics file
    train_set = XGazeH5Dataset(root_dir=root_dir, split="train", training=True, exclude_subjects=val_subjects, demographics_csv_path=train_demog_path, **kwargs)
    val_set = XGazeH5Dataset(root_dir=root_dir, split="train", training=False, include_subjects=val_subjects, demographics_csv_path=train_demog_path, **kwargs)
    
    test_set = None
    try:
        test_set = XGazeH5Dataset(root_dir=root_dir, split="test", training=False, demographics_csv_path=test_demog_path, **kwargs)
    except (KeyError, FileNotFoundError):
        warnings.warn("Could not create a 'test' loader. Trying 'test_person_specific'.")
        try:
             test_set = XGazeH5Dataset(root_dir=root_dir, split="test_person_specific", training=False, demographics_csv_path=test_demog_path, strict=False, **kwargs)
        except (KeyError, FileNotFoundError):
             warnings.warn("No valid test set found.")

    common = _build_common_loader_kwargs(num_workers)
    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size_eval, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size_eval, shuffle=False, drop_last=False, **common) if test_set else None

    return {"train": train_loader, "val": val_loader, "test": test_loader}


