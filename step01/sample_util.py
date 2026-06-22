# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard library imports
import glob
import io
import os
import tarfile as _tarfile
from collections import defaultdict
from typing import Dict

# Third-party imports
import webdataset as wds
from transformers import AutoProcessor

# Define processor globally (assumed to be initialized elsewhere in actual code)
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")


def preprocess_sample(sample: Dict) -> Dict:
    """Preprocess a single raw sample from the WebDataset.

    This function loads the waveform from the raw bytes using torchaudio,
    extracts features using the processor's feature extractor, and tokenizes
    the transcript text.

    Args:
        sample (Dict): A dictionary containing keys 'audio' (raw FLAC bytes)
            and 'text' (transcript string or bytes).

    Returns:
        Dict: A dictionary with keys:
            - 'input_values': processed audio feature tensor.
            - 'labels': list of token IDs corresponding to the transcript.
    """
    # TODO Implement this function
    import soundfile as _sf

    # Decode raw FLAC bytes → float32 numpy array
    audio_bytes = sample["audio"]
    waveform, sample_rate = _sf.read(io.BytesIO(audio_bytes), dtype="float32")

    # Collapse to mono if multi-channel
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Extract wav2vec2 input features via the processor
    input_values = processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
    ).input_values.squeeze(0)

    # Transcript: normalise to str (already uppercase in this dataset)
    text = sample["text"]
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    text = text.strip()

    # Tokenise transcript → list of token IDs
    labels = processor.tokenizer(text).input_ids
    # End of TODO
    return {"input_values": input_values, "labels": labels}


def _tar_shard_generator(data_dir: str):
    """Yield preprocessed samples by reading shard tar files directly.

    This implementation uses Python's tarfile module to avoid WebDataset's
    gopen URL-parsing issues on Windows paths with non-ASCII characters.
    """
    shards = sorted(glob.glob(os.path.join(data_dir, "shard-*.tar")))
    for shard_path in shards:
        with _tarfile.open(shard_path, "r") as tf:
            # Group file members by their base key (strip extension)
            groups: dict = defaultdict(dict)
            for member in tf.getmembers():
                if "." in member.name:
                    base, ext = member.name.rsplit(".", 1)
                    raw = tf.extractfile(member)
                    if raw is not None:
                        groups[base][ext] = raw.read()
            for base in sorted(groups):
                files = groups[base]
                if "audio" not in files or "text" not in files:
                    continue
                text = files["text"]
                if isinstance(text, bytes):
                    text = text.decode("utf-8").strip()
                sample = {
                    "audio": files["audio"],
                    "text": text,
                    "meta": files.get("meta", b""),
                }
                yield preprocess_sample(sample)


class _IterableDataset:
    """Thin iterable wrapper so make_dataset returns a re-iterable object."""

    def __init__(self, data_dir: str):
        self._data_dir = data_dir

    def __iter__(self):
        return _tar_shard_generator(self._data_dir)

    # Support .select() used in train_local.py (no-op passthrough)
    def select(self, fn):
        return self


def make_dataset(data_dir: str):
    """Create a dataset pipeline that loads and preprocesses data shards.

    Reads all shards named 'shard-*.tar' in the given directory using
    Python's tarfile module (works with non-ASCII paths on Windows).

    Args:
        data_dir (str): Path to the directory containing dataset shards.

    Returns:
        Iterable yielding dicts with 'input_values' and 'labels'.
    """
    return _IterableDataset(data_dir)
