import numpy as np
import torch
from typing import Any, Dict, List, Tuple
from transformers import AutoTokenizer

from src.reward_utils import compute_reward

SYSTEM_MESSAGE = ("You are a helpful assistant. You first think about the reasoning process in the mind "
                  "and then provide the user with the answer.")
PROMPT_TEMPLATE = ("Using the numbers {numbers}, create an equation that equals {target}. "
                   "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
                   "Show your work in <think> </think> tags. And return the final equation and answer in "
                   "<answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>.")

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."
DEFAULT_PROMPT_TEMPLATE = "Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."


def create_prompt(
    numbers: List[int],
    target: int,
    tokenizer: AutoTokenizer,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> str:
    prefix = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": prompt_template.format(numbers=numbers, target=target),
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        },
    ]
    return tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True)


def prepare_model_inputs(
    query_token_ids: List[List[int]],
    response_token_ids: List[List[int]],
    advantages: List[List[float]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Prepare padded model inputs with attention masks, labels, and advantages.
    Args:
        query_token_ids: List of query token ids
        response_token_ids: List of response token ids
        advantages: List of lists of advantage values, matching response_token_ids structure
        device: Device to move the tensors to
    Returns:
        Dict with input_ids, attention_mask, labels, and advantages

    Example:
        >>> query_token_ids = [[1, 2, 3], [4, 5]]
        >>> response_token_ids = [[6, 7], [8]]
        >>> advantages = [[0.5, 0.8], [0.3]]
        >>> outputs = prepare_model_inputs(query_token_ids, response_token_ids, advantages, "cuda")
        >>> outputs
        {
            'input_ids': tensor([
                [1, 2, 3, 6, 7],
                [4, 5, 8, 0, 0]
            ]),
            'attention_mask': tensor([
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0]
            ]),
            'labels': tensor([
                [-100, -100, -100, 6, 7],
                [-100, -100, 8, -100, -100]
            ]),
            'advantages': tensor([
                [0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.9, 0.0]
            ])
        }
    """
    max_seq_len = max(len(q) + len(r) for q, r in zip(query_token_ids, response_token_ids))
    inputs = {"input_ids": [], "attention_mask": [], "labels": [], "advantages": [], "labels_mask": []}

    pad_token_id = 0  # Doesn't matter, will be masked
    ignore_index = -100

    for query, response, advantage in zip(query_token_ids, response_token_ids, advantages):
        combined_ids = query + response
        seq_len = len(combined_ids)

        # Create padded sequences
        input_ids = combined_ids + [pad_token_id] * (max_seq_len - seq_len)
        attention_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
        labels = [ignore_index] * len(query) + response + [ignore_index] * (max_seq_len - seq_len)
        labels_mask = [0] * len(query) + [1] * len(response) + [0] * (max_seq_len - seq_len)
        advantages_seq = [0.0] * len(query) + advantage + [0.0] * (max_seq_len - seq_len)

        assert len(input_ids) == max_seq_len
        assert len(attention_mask) == max_seq_len
        assert len(labels) == max_seq_len
        assert len(advantages_seq) == max_seq_len
        assert len(labels_mask) == max_seq_len

        inputs["input_ids"].append(input_ids)
        inputs["attention_mask"].append(attention_mask)
        inputs["labels"].append(labels)
        inputs["advantages"].append(advantages_seq)
        inputs["labels_mask"].append(labels_mask)

    # Convert to tensors
    return {k: torch.tensor(v, dtype=torch.long if k != "advantages" else torch.float, device=device) for k, v in inputs.items()}


# Load and process dataset
def preprocess_example(example: Dict[str, Any], tokenizer: AutoTokenizer):
    numbers: List[int] = example["nums"]
    target: int = example["target"]

    prefix = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE
        },
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(numbers=numbers, target=target)
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        },
    ]
    input_ids = tokenizer.apply_chat_template(prefix, tokenize=True, continue_final_message=True)
    prompt = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {"prompt": prompt, "input_ids": input_ids}


def create_training_episodes(
    samples: List[Dict[str, Any]],
    all_generations: List[List[int]],
    all_finish_reasons: List[str],
    tokenizer: AutoTokenizer,
    generations_per_sample: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process model generations and calculate rewards for training episodes.

    This function processes generated responses and calculates rewards for training episodes by:
    1. Grouping generations by sample (GENERATIONS_PER_SAMPLE responses per input)
    2. Computing rewards and advantages for each response
    3. Processing response tokens

    Args:
        samples: List of input samples, each containing:
            - input_ids: List[int], tokenized input prompt
            - nums: List[int], numbers to use in equation
            - target: int, target value for equation
        all_generations: List of token ID sequences for each generated response
        all_finish_reasons: List of finish reasons for each generation ("stop" or other)

    Returns:
        Tuple containing:
        1. Dictionary with processed data for training:
            - all_query_token_ids: List[List[int]], input token IDs repeated for each generation
            - all_response_token_ids: List[List[int]], response token IDs with EOS tokens added
            - all_advantages: List[List[float]], advantage values repeated for each token
        2. Dictionary with generation statistics:
            - response_lengths: List[int], lengths of generated responses
            - rewards: List[float], raw reward values
            - non_stop_rate: List[bool], whether each generation ended naturally
            - reward_metrics/*: Various reward component metrics

    Example:
        >>> samples = [{"input_ids": [1,2,3], "nums": [1,2,3], "target": 6}]
        >>> generations = [[4,5, EOS_TOKEN_ID], [6,7], [8,9, EOS_TOKEN_ID]]  # 3 generations per sample
        >>> finish_reasons = ["stop", "length", "stop"]
        >>> episodes, stats = create_training_episodes(samples, generations, finish_reasons)
        >>> episodes
        {
            'all_query_token_ids': [[1,2,3], [1,2,3], [1,2,3]],
            'all_response_token_ids': [[4,5,EOS_TOKEN_ID], [6,7], [8,9,EOS_TOKEN_ID]],
            'all_advantages': [[0.5,0.5,0.5], [-1.0,-1.0], [0.5,0.5,0.5]]
        }
    """
    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * generations_per_sample

    # Process responses and calculate rewards
    groups = [list(range(i, i + generations_per_sample)) for i in range(0, len(all_generations), generations_per_sample)
             ]  # example: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    all_query_token_ids, all_responses_token_ids, all_advantages = [], [], []

    stats = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    for sample, group_indices in zip(samples, groups):
        finish_reasons = [all_finish_reasons[i] for i in group_indices]
        response_token_ids = [all_generations[i] for i in group_indices]
        responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)

        rewards_and_metrics = [compute_reward(resp, sample, eos_token=tokenizer.eos_token) for resp in responses]
        rewards, reward_metrics = zip(*rewards_and_metrics)

        rewards = np.array(rewards)  # [group_size]
        response_advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        advantages = [[resp_adv] * len(resp) for resp_adv, resp in zip(response_advantages, response_token_ids)]

        all_query_token_ids.extend([sample["input_ids"]] * generations_per_sample)
        all_responses_token_ids.extend(response_token_ids)
        all_advantages.extend(advantages)

        stats["rewards"].extend(rewards)
        stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
        stats["response_lengths"].extend([len(ids) for ids in response_token_ids])
        for rm in reward_metrics:
            for k, v in rm.items():
                stats.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
        "all_advantages": all_advantages,
    }

    return episodes, stats


if __name__ == "__main__":
    case_0 = {
        "sample": {
            "input_ids": [1, 2, 3],
            "nums": [1, 2, 3],
            "target": 6
        },
        "generations": [[4, 5, 22, 33], [6, 7], [8, 9, 11], [10, 11]],
        "finish_reasons": ["stop", "length", "stop", "stop"]
    }

    case = case_0
    episodes, stats = create_training_episodes([case["sample"]], case["generations"], case["finish_reasons"])
    episodes
