import gzip
import json
from argparse import ArgumentParser
from itertools import repeat, takewhile
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Generator

import lightning.pytorch as pl
import torch
import tqdm
import transformers
from torch.utils.data import IterableDataset, DataLoader


class Dataset(IterableDataset):
    """Generic dataset for pre-tokenized text

    Args:
        Args:
            token_ids_path (str): Path gzipped jsonl file containing token ids.
            max_length (int, optional): Maximum sequence length. Defaults to None.
    """

    def __init__(self, token_ids_path: Path, max_length: Optional[int]) -> None:
        super().__init__()
        self.token_ids_path = token_ids_path
        self.max_length = max_length

    def __iter__(self) -> Iterator[List[int]]:
        with gzip.open(self.token_ids_path) as file:
            for line in file:
                token_ids = json.loads(line)
                if self.max_length is not None and len(token_ids) > self.max_length:
                    token_ids = token_ids[: self.max_length - 1] + token_ids[-1:]
                yield token_ids


class DataModule(pl.LightningDataModule):
    """Generic DataModule for pre-tokenized text

    Args:
        token_ids_path (str): Path gzipped jsonl file containing token ids.
        pad_token_id (int): ID of padding token in vocabulary.
        sep_token_id (int): ID of separator token in vocabulary.
        max_length (int, optional): Maximum sequence length. Defaults to None.
        batch_size (int, optional): Batch size for training. Defaults to 1.
        shuffle (bool, optional): Toggle for shuffling dataset. Defaults to True.
    """

    def __init__(
        self,
        token_ids_path: Path,
        pad_token_id: int,
        max_length: Optional[int] = None,
        batch_size: int = 1,
    ):
        super().__init__()
        self.token_ids_path = Path(token_ids_path)
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = Dataset(self.token_ids_path, self.max_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate,
        )

    def _collate(self, batch: List[List[int]]) -> transformers.BatchEncoding:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(elem) for elem in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = (input_ids != self.pad_token_id).long()
        encoding = transformers.BatchEncoding(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        return encoding


def _count_lines(path: Path) -> int:
    """Counts number of lines in a file efficiently

    Args:
        path (Path): File path to count lines

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


def batch_tokenize(
    file_path: Path,
    tokenizer_name: str,
    batch_size: int,
    num_lines: Optional[int],
) -> Generator[List[int], None, None]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    with file_path.open("r", encoding="utf8") as fp:
        pg = tqdm.tqdm(fp, total=num_lines)
        texts = []
        for idx, text in enumerate(pg):
            texts.append(text)
            if (idx + 1) % batch_size == 0:
                token_ids = tokenizer(texts).input_ids
                yield from token_ids
                texts = []
        token_ids = tokenizer(texts).input_ids
        yield from token_ids


def tokenize(
    text_files: List[Path],
    save_path: Path,
    tokenizer_name: str,
    batch_size: int = 1000,
    num_lines: bool = False,
) -> None:
    """Tokenize text files and writes the token ids to a gzipped jsonl file. Each line in a
    text file is considered as a new sample.

    Args:
        text_files (List[Path]): Text file paths to tokenize
        save_path (Path): Path to save the pickled token ids to
        tokenizer_name (str): Name of or path to local huggingface tokenizer
        batch_size (int, optional): Number of lines to tokenize at once. Defaults to 1000.
        num_lines (bool, optional): Set to True to count lines before tokenizing. Defaults to False.
    """

    total = None
    if num_lines:
        total = 0
        for text_file in text_files:
            total += _count_lines(text_file)

    token_ids = []
    with gzip.open(save_path, "wb") as file:
        for text_file in text_files:
            for token_ids in batch_tokenize(
                text_file, tokenizer_name, batch_size, num_lines
            ):
                file.write((json.dumps(token_ids) + "\n").encode("utf8"))


def main(cli_args: Optional[Sequence[str]] = None):
    parser = ArgumentParser(
        description="Tokenizes text files and writes the token ids to a pickle file. "
        "Each line in a text file is considered as a new sample. Optionally splits "
        "the input by sentences. "
    )

    parser.add_argument(
        "--text_files",
        type=Path,
        nargs="+",
        required=True,
        help="Text file paths to tokenize",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
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
        "--batch_size",
        type=int,
        default=1000,
        help="Number of lines to tokenize at once.",
    )
    parser.add_argument(
        "--num_lines",
        action="store_true",
        help="Count number of lines before tokenizing. Useful for estimating ETA.",
    )

    args = parser.parse_args(cli_args)

    tokenize(**vars(args))


if __name__ == "__main__":
    main()
