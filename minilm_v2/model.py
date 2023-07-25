import math
from pathlib import Path
from typing import Any, Dict

import lightning.pytorch as pl
import lightning.pytorch.utilities as pl_utilities
import torch
import torch.utils.data
import transformers


class MiniLMV2Module(pl.LightningModule):
    def __init__(
        self,
        teacher_model_name_or_path: str,
        student_model_name_or_path: str,
        teacher_layer_index: int = -1,
        student_layer_index: int = -1,
        num_relation_heads: int = -1,
    ):
        super().__init__()
        self.teacher = transformers.AutoModel.from_pretrained(
            teacher_model_name_or_path
        )
        student_config = transformers.AutoConfig.from_pretrained(
            student_model_name_or_path
        )
        self.student = transformers.AutoModel.from_config(student_config)

        self.student_model_name_or_path = student_model_name_or_path

        self.num_relation_heads = num_relation_heads

        self.cache: Dict[str, torch.Tensor] = {}
        self._cache_qkv(self.teacher, teacher_layer_index, "teacher")
        self._cache_qkv(self.student, student_layer_index, "student")

        self._validate_model_dims(num_relation_heads)

    def _validate_model_dims(self, num_relation_heads: int) -> None:
        student_hidden_size = self.student.config.hidden_size
        student_attention_heads = self.student.config.num_attention_heads
        teacher_hidden_size = self.teacher.config.hidden_size
        teacher_attention_heads = self.teacher.config.num_attention_heads

        if student_hidden_size % student_attention_heads != 0:
            raise ValueError(
                f"student hidden_size ({student_hidden_size}) must be multiple "
                f"of num_attention_heads ({student_attention_heads})"
            )
        if teacher_hidden_size % teacher_attention_heads != 0:
            raise ValueError(
                f"teacher hidden_size ({teacher_hidden_size}) must be multiple "
                f"of num_attention_heads ({teacher_attention_heads})"
            )
        if student_hidden_size % num_relation_heads != 0:
            raise ValueError(
                f"student hidden_size ({student_hidden_size}) must be multiple "
                f"of num_relation_heads ({num_relation_heads})"
            )
        if teacher_hidden_size % num_relation_heads != 0:
            raise ValueError(
                f"teacher hidden_size ({teacher_hidden_size}) must be multiple "
                f"of num_relation_heads ({num_relation_heads})"
            )
        if (
            num_relation_heads <= 0
            and teacher_attention_heads != student_attention_heads
        ):
            ValueError(
                "student and teacher attention heads must be equal, "
                "use num_relation_heads to equalize dimensions"
            )

    def training_step(self, data_batch, batch_i):
        with torch.no_grad():
            self.teacher(**data_batch)
        self.student(**data_batch)
        loss = self._aggregrate_minilm_loss(
            data_batch["attention_mask"], self.num_relation_heads
        )
        self.log("loss", loss, prog_bar=True)
        return loss

    def _minilm_loss(
        self,
        student_tensor: torch.Tensor,
        teacher_tensor: torch.Tensor,
        attention_mask: torch.Tensor,
        num_relation_heads: int,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = student_tensor.shape
        student_tensor = student_tensor.reshape(
            batch_size, seq_len, self.student.config.num_attention_heads, -1
        )
        teacher_tensor = teacher_tensor.reshape(
            batch_size, seq_len, self.teacher.config.num_attention_heads, -1
        )
        if num_relation_heads > 0:
            student_tensor = student_tensor.reshape(
                batch_size, seq_len, num_relation_heads, -1
            )
            teacher_tensor = teacher_tensor.reshape(
                batch_size, seq_len, num_relation_heads, -1
            )
        student_tensor = student_tensor.permute(0, 2, 1, 3)
        teacher_tensor = teacher_tensor.permute(0, 2, 1, 3)

        student_head_dim = student_tensor.shape[3]
        teacher_head_dim = teacher_tensor.shape[3]

        student_dot_product = student_tensor.matmul(student_tensor.transpose(-1, -2))
        student_scaled_dot_product = student_dot_product / math.sqrt(student_head_dim)
        student_scaled_dot_product = student_scaled_dot_product + attention_mask
        student_probs = torch.nn.functional.log_softmax(
            student_scaled_dot_product, dim=-1
        )

        teacher_dot_product = teacher_tensor.matmul(teacher_tensor.transpose(-1, -2))
        teacher_scaled_dot_product = teacher_dot_product / math.sqrt(teacher_head_dim)
        teacher_scaled_dot_product = teacher_scaled_dot_product + attention_mask
        teacher_probs = torch.nn.functional.log_softmax(
            teacher_scaled_dot_product, dim=-1
        )

        loss = torch.nn.functional.kl_div(
            student_probs,
            teacher_probs,
            reduction="sum",
            log_target=True,
        ) / (batch_size * seq_len * num_relation_heads)
        return loss

    def _aggregrate_minilm_loss(
        self, attention_mask: torch.Tensor, num_relation_heads: int = -1
    ) -> torch.Tensor:
        device = self.cache["student_key"].device
        if num_relation_heads == -1:
            num_relation_heads = self.student.config.num_attention_heads
        attention_mask = self.teacher.get_extended_attention_mask(
            attention_mask, attention_mask.shape, device
        )
        loss = torch.tensor(0.0, requires_grad=True, device=device)
        for element in ["key", "query", "value"]:
            loss = loss + self._minilm_loss(
                self.cache[f"student_{element}"],
                self.cache[f"teacher_{element}"],
                attention_mask,
                num_relation_heads,
            )
        loss = loss / 3
        return loss

    def _cache_variable(
        self, module: torch.nn.Module, variable: str
    ) -> torch.nn.Module:
        def _forward(*args, **kwargs):
            out = module._forward(*args, **kwargs)
            self.cache[variable] = out
            return out

        module._forward = module.forward
        module.forward = _forward
        return module

    def _cache_qkv(
        self, model: torch.nn.Module, layer_index: int, variable_name: str
    ) -> None:
        if layer_index == -1:
            layer_index = model.config.num_hidden_layers - 1

        for module_name, module in model.named_modules():
            if f"encoder.layer.{layer_index}.attention.self." in module_name:
                name = module_name.split(".")[-1]
                if name in ("query", "key", "value"):
                    self._cache_variable(module, f"{variable_name}_{name}")

    def save_pretrained(self, path: Path) -> None:
        self.student.save_pretrained(path)

    @pl_utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.trainer is not None:
            if Path(self.student_model_name_or_path).exists():
                save_path = Path(self.student_model_name_or_path)
            elif self.trainer.log_dir is not None:
                save_path = Path(self.trainer.log_dir) / "huggingface_checkpoint"
            else:
                save_path = Path.cwd() / "huggingface_checkpoint"
            self.save_pretrained(save_path)
