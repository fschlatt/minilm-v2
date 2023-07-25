from minilm_v2 import MiniLMV2Module
from pathlib import Path
from transformers import AutoTokenizer

CONFIG_DIR = Path(__file__).parent.parent / "configs"
TEST_DATA_DIR = Path(__file__).parent / "data"


def test_training_step():
    l6_h384_config = CONFIG_DIR / "l6-h384"
    module = MiniLMV2Module("bert-base-uncased", str(l6_h384_config), -1, -1, 48)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_text = (TEST_DATA_DIR / "test_text.txt").read_text().split("\n")
    encoding = tokenizer(test_text, return_tensors="pt", padding=True)
    loss = module.training_step(encoding, 0)
    loss.backward()
    assert loss.item() > 0
    assert loss.requires_grad
    assert module.student.encoder.layer[-1].output.dense.weight.grad is None
    assert module.student.encoder.layer[-2].output.dense.weight.grad is not None
    assert module.teacher.encoder.layer[-2].output.dense.weight.grad is None
