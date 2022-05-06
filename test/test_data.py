import pathlib
import pickle

import minilm_v2.data
import pytest

CONTENT = """This is one sentence. This a second sentence.\nThis is a new line."""


@pytest.fixture(scope="session")
def tmp_text_file(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    tmp_text_file = tmp_path_factory.mktemp("minilm_v2_data") / "test_text.txt"
    tmp_text_file.write_text(CONTENT)
    return tmp_text_file


@pytest.fixture(scope="session")
def tmp_tokenized_file(tmp_text_file: pathlib.Path) -> pathlib.Path:
    save_path = tmp_text_file.with_name("test_text.pkl")

    minilm_v2.data.tokenize([tmp_text_file], save_path, "bert-base-uncased", False)

    return save_path


def test_tokenize(tmp_text_file: pathlib.Path) -> None:

    save_path = tmp_text_file.with_name("test_text.pkl")

    minilm_v2.data.tokenize([tmp_text_file], save_path, "bert-base-uncased", False)
    with save_path.open("rb") as file:
        data = pickle.load(file)
    assert len(data) == 2

    minilm_v2.data.tokenize([tmp_text_file], save_path, "bert-base-uncased", True)
    with save_path.open("rb") as file:
        data = pickle.load(file)
    assert len(data) == 3


def test_main(tmp_text_file: pathlib.Path) -> None:

    save_path = tmp_text_file.with_name("test_text.pkl")
    args = [
        "--text_files",
        str(tmp_text_file),
        "--save_path",
        str(save_path),
        "--tokenizer_name",
        "bert-base-uncased",
    ]

    minilm_v2.data.main(args)
    with save_path.open("rb") as file:
        data = pickle.load(file)
    assert len(data) == 2


def test_dataset(tmp_tokenized_file: pathlib.Path) -> None:
    dataset = minilm_v2.data.Dataset(tmp_tokenized_file)
    assert len(dataset) == 2
    assert dataset[0].shape != dataset[1].shape


def test_datamodule(tmp_tokenized_file: pathlib.Path) -> None:
    datamodule = minilm_v2.data.Datamodule(
        tmp_tokenized_file, pad_token_id=0, sep_token_id=102, batch_size=2
    )
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    data = next(iter(dataloader))
    assert len(dataloader) == 1
    assert any(sum(data.input_ids == 0))

    datamodule.max_length = 5
    data = next(iter(dataloader))
    assert data.input_ids.shape[1] == 5
