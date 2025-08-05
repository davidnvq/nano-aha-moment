from dataclasses import dataclass
from typing import Any, Dict, Tuple
from vllm import SamplingParams


@dataclass
class TrainingArguments:
    """Training arguments for R1 training with PPO."""

    # Model configuration
    model_name: str = "Qwen/Qwen2.5-0.5B"
    model_chat_name: str = None  # Will be set to model_name + "-Instruct"

    # Dataset configuration
    dataset_name: str = "Jiayi-Pan/Countdown-Tasks-3to4"

    # Training iterations
    num_iterations: int = 1000
    episodes_per_iteration: int = 64
    generations_per_sample: int = 4  # Group size in GRPO

    # RL parameters
    kl_coefficient: float = 0.001  # Controls how much policy can deviate from reference model

    # Training hyperparameters
    per_device_batch_size: int = 4
    learning_rate: float = 1e-6

    # Sampling parameters
    max_response_tokens: int = 1024
    temperature: float = 1.0  # Controls randomness in generation
    top_p: float = 1.0  # Nucleus sampling parameter (1.0 = disabled)
    top_k: int = -1  # Top-k sampling parameter (-1 = disabled)

    # DeepSpeed configuration
    deepspeed_stage: int = 2
    gradient_clipping: float = 1.0
    optimizer_betas: Tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    weight_decay: float = 0.0

    # vLLM configuration
    gpu_memory_utilization: float = 0.4  # 0.4 of 25GB = 10GB
    swap_space: int = 2  # GB
    max_model_len: int = 2048

    # Logging and checkpointing
    run_name: str = "r1-zero"
    eval_interval: int = 25
    checkpoint_interval: int = 50

    def __post_init__(self):
        """Set derived values after initialization."""
        if self.model_chat_name is None:
            self.model_chat_name = self.model_name + "-Instruct"

    def get_deepspeed_config(self) -> Dict[str, Any]:
        """Get DeepSpeed configuration for policy model."""
        return {
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": self.deepspeed_stage,
                "overlap_comm": False
            },
            "train_batch_size": self.episodes_per_iteration,
            "train_micro_batch_size_per_gpu": self.per_device_batch_size,
            "gradient_accumulation_steps": self.episodes_per_iteration // self.per_device_batch_size,
            "gradient_clipping": self.gradient_clipping,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.learning_rate,
                    "betas": self.optimizer_betas,
                    "eps": self.optimizer_eps,
                    "weight_decay": self.weight_decay,
                    "torch_adam": True,
                },
            },
        }

    def get_ref_deepspeed_config(self) -> Dict[str, Any]:
        """Get DeepSpeed configuration for reference model."""
        return {
            "bf16": {
                "enabled": True
            },
            # Note that we don't train the reference model
            # These are just for compatibility with DeepSpeed.
            "train_batch_size": self.episodes_per_iteration,
            "train_micro_batch_size_per_gpu": self.per_device_batch_size,
            "gradient_accumulation_steps": self.episodes_per_iteration // self.per_device_batch_size,
        }

    def get_sampling_params(self, eos_token_id: int) -> SamplingParams:
        """Get vLLM sampling parameters."""
        return SamplingParams(
            n=self.generations_per_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_response_tokens,
            detokenize=False,
            stop_token_ids=[eos_token_id],
        )

    def get_eval_sampling_params(self, eos_token_id: int) -> SamplingParams:
        """Get vLLM sampling parameters for evaluation."""
        return SamplingParams(
            temperature=0.3,
            max_tokens=1024,
            n=1,
            detokenize=False,
            stop_token_ids=[eos_token_id],
        )


if __name__ == "__main__":
    from rich import print
    from transformers import HfArgumentParser
    args = HfArgumentParser(TrainingArguments).parse_args_into_dataclasses()[0]
    print(args)
