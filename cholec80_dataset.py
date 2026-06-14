import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image

CHOLEC80_TOOLS = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag']
CHOLEC80_NUM_CLASSES = len(CHOLEC80_TOOLS)

TRAIN_VIDEOS = [f'{i:02d}' for i in range(1, 41)]
VAL_VIDEOS = [f'{i:02d}' for i in range(41, 61)]
TEST_VIDEOS = [f'{i:02d}' for i in range(61, 81)]


class Cholec80Dataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        self.frames_dir = self.data_dir / 'frames_1fps'
        self.annot_dir = self.data_dir / 'tool_annotations'

        if split == 'train':
            self.video_ids = TRAIN_VIDEOS
        elif split == 'val':
            self.video_ids = VAL_VIDEOS
        elif split == 'test':
            self.video_ids = TEST_VIDEOS
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")

        self.samples = []
        for vid in self.video_ids:
            annot_file = self.annot_dir / f'video{vid}-tool.txt'
            with open(annot_file, 'r') as f:
                lines = f.readlines()

            frame_to_label = {}
            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) < 8:
                    continue
                frame_idx = int(parts[0])
                label = np.array([int(x) for x in parts[1:8]], dtype=np.float32)
                frame_to_label[frame_idx] = label

            video_frame_dir = self.frames_dir / f'video{vid}'
            if not video_frame_dir.exists():
                continue

            png_files = sorted(video_frame_dir.glob('*.png'))
            for png_file in png_files:
                png_num = int(png_file.stem)
                frame_idx = (png_num - 1) * 25
                if frame_idx in frame_to_label:
                    self.samples.append((str(png_file), frame_to_label[frame_idx]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(label).float()
