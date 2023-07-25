from pathlib import Path
import gzip
import json

import minilm_v2.data
import pytest


TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def datamodule() -> minilm_v2.data.DataModule:
    datamodule = minilm_v2.data.DataModule(
        TEST_DATA_DIR / "test_text.jsonl.gz", pad_token_id=0, batch_size=2
    )
    datamodule.setup()
    return datamodule


def test_tokenize(tmp_path_factory: pytest.TempPathFactory) -> None:
    save_path = tmp_path_factory.mktemp("test_tokenize") / "test_text.jsonl.gz"
    text_file = TEST_DATA_DIR / "test_text.txt"

    minilm_v2.data.tokenize([text_file], save_path, "bert-base-uncased")
    with gzip.open(save_path) as file:
        data = [json.loads(line) for line in file]
    assert len(data) == 2


def test_main(tmp_path_factory: pytest.TempPathFactory) -> None:
    save_path = tmp_path_factory.mktemp("test_tokenize") / "test_text.jsonl.gz"
    text_file = TEST_DATA_DIR / "test_text.txt"
    args = [
        "--text_files",
        str(text_file),
        "--save_path",
        str(save_path),
        "--tokenizer_name",
        "bert-base-uncased",
    ]

    minilm_v2.data.main(args)
    with gzip.open(save_path) as file:
        data = [json.loads(line) for line in file]
    assert len(data) == 2


def test_dataset() -> None:
    tokenized_file = TEST_DATA_DIR / "test_text.jsonl.gz"
    dataset = minilm_v2.data.Dataset(tokenized_file, max_length=None)
    num_lines = 0
    for _ in dataset:
        num_lines += 1
    assert num_lines == 2


def test_datamodule(datamodule: minilm_v2.data.DataModule) -> None:
    dataloader = datamodule.train_dataloader()
    data = next(iter(dataloader))
    assert any(sum(data.input_ids == 0))

    datamodule.train_dataset.max_length = 5
    data = next(iter(dataloader))
    assert data.input_ids.shape[1] == 5
