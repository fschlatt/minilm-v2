import argparse
import datetime
import logging
import pathlib
import pickle
import time
from itertools import repeat, takewhile
from typing import List, Optional, Sequence

import nltk
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import tqdm
import transformers

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Datamodule(pl.LightningDataModule):
    """Generic DatamModule for pretokenized text

    Args:
        token_ids_path (str): Path to pickled token_ids file
        tokenizer (str): Path to huggingface or local tokenizer
        batch_size (int, optional): Batch size for training. Defaults to 1.
        shuffle (bool, optional): Toggle for shuffling dataset. Defaults to True.
    """

    def __init__(
        self,
        token_ids_path: pathlib.Path,
        pad_token_id: int,
        batch_size: int = 1,
        shuffle: bool = True,
    ):
        super().__init__()
        self.token_ids_path = pathlib.Path(token_ids_path)
        self.pad_token_id = pad_token_id
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = Dataset(self.token_ids_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self._collate,
        )

    def _collate(self, batch: List[np.ndarray]) -> transformers.BatchEncoding:
        tensor_list = [torch.tensor(elem) for elem in batch]
        padding_value = self.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            tensor_list, batch_first=True, padding_value=padding_value
        )
        attention_mask = (input_ids != padding_value).long()
        encoding = transformers.BatchEncoding(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        return encoding


class Dataset(torch.utils.data.Dataset):
    """Generic dataset for pretoknized text

    Args:
        Args:
            token_ids_path (str): Path to pickled token_ids file
    """

    def __init__(self, token_ids_path: pathlib.Path) -> None:
        super().__init__()
        with token_ids_path.open("rb") as file:
            self.data = pickle.load(file)

    def __getitem__(self, index: int) -> np.ndarray:
        token_ids = self.data[index].astype(np.dtype("int64"))
        return token_ids

    def __len__(self) -> int:
        return len(self.data)


def _count_lines(path: pathlib.Path) -> int:
    """Counts number of lines in a file efficiently

    Args:
        path (pathlib.Path): File path to count lines

    Returns:
        int: Number of lines in the file
    """
    bufgen = takewhile(
        lambda x: x, (path.open("rb").read(1024 * 1024) for _ in repeat(None))
    )
    count = 0
    for buf in bufgen:
        count += buf.count(b"\n")
    return count


def tokenize(
    text_files: List[pathlib.Path],
    save_path: pathlib.Path,
    tokenizer_name: str,
    split_sentences: bool,
    batch_size: int = 1000,
    num_lines: Optional[int] = None,
) -> None:
    """Tokenize text files and writes the token ids to a pickle file. Each line in a
    text file is considered as a new sample. Optionally splits the input by sentences.

    Args:
        text_files (List[pathlib.Path]): Text file paths to tokenize
        save_path (pathlib.Path): Path to save the pickled token ids to
        tokenizer_name (str): Name of or path to local huggingface tokenizer
        split_sentences (bool): Toggle to split text into sentences
        batch_size (int, optional): Number of lines to tokenize at once.
        Defaults to 1000.
        num_lines (Optional[int], optional): Total number of lines across all text 
        files. Used to estimate total tokenization time. Set to -1 automatically count 
        the number of lines. Defaults to None.
    """

    if num_lines == -1:
        logger.info("Counting number of lines.")
        num_lines = 0
        for text_file in text_files:
            num_lines += _count_lines(text_file)

    logger.info(f"Loading Tokenizer ({tokenizer_name}).")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

    vocab_size = tokenizer.vocab_size
    logger.info("Tokenizing text.")

    splitter = nltk.sent_tokenize if split_sentences else lambda x: [x]

    token_ids = []
    pg = tqdm.tqdm(total=num_lines)
    num_samples = 0
    start = time.perf_counter()
    for text_file in text_files:
        with text_file.open("r", encoding="utf8") as file:
            texts = []
            for idx, text in enumerate(file):
                texts.extend(splitter(text))
                if (idx + 1) % batch_size == 0:
                    token_ids.extend(tokenizer(texts)["input_ids"])
                    num_samples += len(texts)
                    texts = []
                pg.update()
            if texts:
                token_ids.extend(tokenizer(texts)["input_ids"])
                num_samples += len(texts)
    elapsed = time.perf_counter() - start

    logger.info(
        f"Tokenized {num_lines} in {datetime.timedelta(seconds=elapsed)} "
        f"lines and obtained {num_samples} samples."
    )

    dtype = np.dtype("uint16") if vocab_size < (1 << 16) else np.dtype("int32")
    logger.info("Converting token_ids to numpy arrays.")
    token_ids = [np.array(token_list, dtype=dtype) for token_list in token_ids]

    logger.info("Saving file.")
    with save_path.open("wb") as file:
        pickle.dump(token_ids, file)


def main(cli_args: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Tokenizes text files and writes the token ids to a pickle file. "
        "Each line in a text file is considered as a new sample. Optionally splits "
        "the input by sentences. "
    )

    parser.add_argument(
        "--text_files",
        type=pathlib.Path,
        nargs="+",
        required=True,
        help="Text file paths to tokenize",
    )
    parser.add_argument(
        "--save_path",
        type=pathlib.Path,
        required=True,
        help="Path to save the pickled token ids to",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Name of or path to local huggingface tokenizer",
    )
    parser.add_argument(
        "--split_sentences",
        action="store_true",
        help="Toggle to split text into sentences",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of lines to tokenize at once.",
    )
    parser.add_argument(
        "--num_lines",
        type=int,
        default=None,
        help="Total number of lines across all text ",
    )

    args = parser.parse_args(cli_args)

    tokenize(**vars(args))


if __name__ == "__main__":
    main()
