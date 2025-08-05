import torch
from typing import Any, Dict, List, Tuple, Union
from deepspeed import DeepSpeedEngine
from transformers import PreTrainedModel


@torch.compile(dynamic=True)
def log_softmax_and_gather(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Copied from https://github.com/allenai/open-instruct/blob/main/open_instruct/model_utils.py#L425

    torch compiled version of the common `log_softmax -> gather` operation.

    The compiled version of this opration avoids the (significant) memory overhead of
    allocating a new (batch_size, seq_len, vocab_size) tensor to store the logprobs.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size) containing the logits
        index: Tensor of shape (batch_size, seq_len) containing the indices to gather

    Returns:
        Tensor of shape (batch_size, seq_len) containing the log probabilities for the
        specified indices

    See https://github.com/allenai/open-instruct/pull/584
    """
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def compute_token_log_probs(
    model: Union[DeepSpeedEngine, PreTrainedModel],
    inputs: Dict[str, torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    """
    Compute log probabilities for each token in the sequence, masked for valid labels only.

    This function:
    1. Runs the model forward pass
    2. Applies temperature scaling to logits
    3. Shifts the sequences for causal language modeling
    4. Computes log probabilities for the actual tokens that appeared in the sequence
    5. Masks the log probabilities to only include valid labels (non -100 positions)

    Args:
        model: The language model (either DeepSpeed-wrapped or regular HuggingFace model)
        inputs: Dictionary containing:
            - input_ids: Tensor of token ids [batch_size, seq_len]
            - attention_mask: Tensor of attention mask [batch_size, seq_len]
            - labels: Tensor of target labels [batch_size, seq_len] with -100 for ignored positions
        temperature: Temperature for scaling the logits before softmax

    Returns:
        torch.Tensor: Log probabilities tensor of shape [batch_size, seq_len-1], where:
            - Each value is the log probability of the actual token that appeared
            - Values are masked to 0.0 for positions where labels were -100
            - The sequence length is reduced by 1 due to the causal shift

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> inputs = {
        ...     "input_ids": torch.tensor([[1, 2, 3]]),
        ...     "attention_mask": torch.tensor([[1, 1, 1]]),
        ...     "labels": torch.tensor([[-100, 2, 3]])
        ... }
        >>> log_probs = compute_token_log_probs(model, inputs, temperature=1.0)
        >>> log_probs.shape
        torch.Size([1, 2])  # batch_size=1, seq_len-1=2
        >>> # First position is 0 (masked), second position has actual log prob
    """
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
        use_cache=False,
    )

    logits = outputs.logits / temperature  # Shape: [batch_size, seq_len, vocab_size]
    shift_logits = logits[..., :-1, :]  # Shape: [batch_size, seq_len-1, vocab_size]
    shift_labels = inputs["labels"][..., 1:]  # Shape: [batch_size, seq_len-1]
    shift_labels_mask = inputs["labels_mask"][..., 1:]  # Shape: [batch_size, seq_len-1]

    # Create mask for valid labels
    shift_labels[~(shift_labels_mask.bool())] = 0  # Shape: [batch_size, seq_len-1]

    # Calculate log probabilities
    log_probs = log_softmax_and_gather(shift_logits, shift_labels)  # Shape: [batch_size, seq_len-1]
    log_probs = log_probs * shift_labels_mask  # Shape: [batch_size, seq_len-1]

    return log_probs


def compute_pg_loss(
    policy_model: Union[DeepSpeedEngine, PreTrainedModel],
    reference_model: Union[DeepSpeedEngine, PreTrainedModel],
    batch: Dict[str, torch.Tensor],
    total_response_len: int,
    temperature: float,
    kl_coefficient: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the policy gradient loss with KL penalty between policy and reference models.

    This function:
    1. Computes log probabilities for both policy and reference models
    2. Calculates KL divergence penalty between the models
    3. Computes policy gradient loss using advantages
    4. Combines the losses with KL coefficient

    Args:
        policy_model: The model being trained
        reference_model: The reference model for KL penalty calculation
        batch: Dictionary containing:
            - input_ids: Tensor of shape [batch_size, seq_len]
            - attention_mask: Tensor of shape [batch_size, seq_len]
            - labels: Tensor of shape [batch_size, seq_len] with -100 for ignored positions
            - advantages: Tensor of shape [batch_size, seq_len]

    Returns:
        Tuple containing:
            - loss: Combined policy gradient and KL penalty loss (scalar tensor)
            - metrics: Dictionary with detailed loss components:
                - policy_loss: Pure policy gradient loss
                - kl_penalty: KL divergence penalty
                - entropy: Policy entropy
    """
    input_ids = batch["input_ids"]  # [batch_size, seq_len]
    attention_mask = batch["attention_mask"]  # [batch_size, seq_len]
    labels = batch["labels"]  # [batch_size, seq_len]
    advantages = batch["advantages"]  # [batch_size, seq_len]

    labels_mask = (labels[..., :] != -100).float()  # [batch_size, seq_len]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "labels_mask": labels_mask,
    }

    with torch.no_grad():
        ref_logps = compute_token_log_probs(reference_model, model_inputs, temperature)  # [batch_size, seq_len-1]

    logps = compute_token_log_probs(policy_model, model_inputs, temperature)  # [batch_size, seq_len-1]

    shifted_labels_mask = labels_mask[..., 1:]  # [batch_size, seq_len-1]

    kl_penalty = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1  # [batch_size, seq_len-1]
    kl_penalty = kl_penalty * shifted_labels_mask  # [batch_size, seq_len-1]

    entropy = -logps.sum() / shifted_labels_mask.sum()  # scalar

    policy_loss = -logps * advantages[..., 1:]  # [batch_size, seq_len-1]
    policy_loss = policy_loss * shifted_labels_mask  # [batch_size, seq_len-1]

    loss = (policy_loss + kl_coefficient * kl_penalty).sum() / total_response_len  # scalar

    metrics = {
        "policy_loss": policy_loss.sum().item() / total_response_len,
        "kl_penalty": kl_penalty.sum().item() / total_response_len,
        "entropy": entropy.item() / total_response_len,
    }

    return loss, metrics
