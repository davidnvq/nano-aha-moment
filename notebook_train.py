import os
import gc
import time
from pathlib import Path
from functools import partial

import torch
import deepspeed
import numpy as np
from tqdm import trange
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM
import wandb

from utils import (dump_episodes, evaluate_on_test_set, find_free_port, find_last_checkpoint, prepare_model_inputs, load_model_into_vllm)

from src.data_utils import preprocess_example, create_training_episodes
from src.reward_utils import compute_reward
from src.grpo_loss import compute_pg_loss
from src.config import TrainingArguments
from transformers import HfArgumentParser

if __name__ == "__main__":

    # Set the environment variables for HuggingFace
    # This is done to ensure that the cache directory for HuggingFace is set to a specific location,
    # preventing the storage from being overwhelmed with model files and other data.
    SCRATCH = Path.home() / "scratch"
    os.environ["HF_HOME"] = str(SCRATCH / "hf_home")
    os.environ["VLLM_USE_V1"] = "0"

    # Needed to stop DeepSpeed from complaining
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # Initialize training arguments
    args = HfArgumentParser(TrainingArguments).parse_args_into_dataclasses()[0]

    deepspeed_config = args.get_deepspeed_config()
    ref_deepspeed_config = args.get_ref_deepspeed_config()

    EXP_DIR = Path("checkpoints") / args.run_name
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Logs and Checkpoints will be saved to: {EXP_DIR}")

    # Note that the base model and "instruct" model have different eos token.
    # Here we make sure to use the correct one.
    tokenizer = AutoTokenizer.from_pretrained(args.model_chat_name)
    EOS_TOKEN_ID = AutoTokenizer.from_pretrained(args.model_name).eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)

    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.map(partial(preprocess_example, tokenizer=tokenizer), num_proc=6)

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

    # Initialize main and reference models
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=0,
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=0,
    )
    policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Initialize DeepSpeed engines
    policy_model, *_ = deepspeed.initialize(
        model=policy_model,
        config=deepspeed_config,
        model_parameters=policy_model.parameters(),
    )
    reference_model, *_ = deepspeed.initialize(
        model=reference_model,
        config=ref_deepspeed_config,
    )

    reference_model.module.cpu()

    ############################################
    # Initialize vLLM (Inference) engine
    ############################################

    inference_engine = LLM(
        model=args.model_name,
        skip_tokenizer_init=False,
        # gpu_memory_utilization=0.2, 0.2 of 80GB = 16GB
        gpu_memory_utilization=args.gpu_memory_utilization,  # 0.4 of 25GB = 10GB
        enable_prefix_caching=True,
        swap_space=args.swap_space,  # 2GB
        scheduling_policy="fcfs",
        dtype=torch.bfloat16,
        max_model_len=args.max_model_len,
        enable_sleep_mode=True,
    )

    # Wandb for logging
    wandb.init(
        project="r1-aha-moment",
        name=args.run_name,
        config={
            "model_name": args.model_name,
            "learning_rate": args.learning_rate,
            "num_iterations": args.num_iterations,
            "episodes_per_iteration": args.episodes_per_iteration,
            "rollouts_per_episode": args.generations_per_sample,
            "kl_coefficient": args.kl_coefficient,
            "temperature": args.temperature,
        },
    )

    # Load checkpoint if it exists
    begin_iter = 0
    ckpt_path, ckpt_iter = find_last_checkpoint(EXP_DIR)
    if ckpt_path is not None:
        print(f"Resuming from checkpoint {ckpt_path} at iteration {ckpt_iter}")
        out = policy_model.load_checkpoint(ckpt_path / "deepspeed")
        if out is None:
            raise RuntimeError(f"Failed to load checkpoint {ckpt_path}")
        begin_iter = ckpt_iter + 1
        load_model_into_vllm(policy_model, inference_engine)

    for iteration in trange(args.num_iterations):
        print(f"Iteration {iteration}/{args.num_iterations}")

        metrics = {}

        #########################################################
        # Evaluation
        #########################################################

        eval_stats = None
        if iteration % args.eval_interval == 0:
            print("Evaluating on eval set...")
            eval_episodes, eval_stats = evaluate_on_test_set(
                inference_engine=inference_engine,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                eos_token=EOS_TOKEN,
                eval_sampling_params=args.get_eval_sampling_params(EOS_TOKEN_ID),
                reward_func=lambda completion, sample: compute_reward(completion, sample, eos_token=EOS_TOKEN),
            )
            eval_episode_table = dump_episodes(
                episodes=eval_episodes,
                episodes_stats=eval_stats,
                exp_dir=EXP_DIR,
                tokenizer=tokenizer,
                iteration=iteration,
                is_eval=True,
            )
            wandb.log({"eval/episodes": eval_episode_table, "iteration": iteration})

        #########################################################
        # Generate Episodes
        #########################################################

        # Sample training batch
        num_samples = args.episodes_per_iteration // args.generations_per_sample
        indices = np.random.choice(len(train_dataset), size=num_samples, replace=False)
        samples = train_dataset.select(indices)

        # Sample responses
        sampling_params = args.get_sampling_params(EOS_TOKEN_ID)
        outputs = inference_engine.generate(prompt_token_ids=samples["input_ids"], sampling_params=sampling_params)
        all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
        all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]
        inference_engine.sleep(1)

        print(f"Generated {len(all_generations)} responses")
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        # Process responses and calculate rewards
        episodes, episodes_stats = create_training_episodes(
            samples,
            all_generations,
            all_finish_reasons,
            tokenizer=tokenizer,
            generations_per_sample=args.generations_per_sample,
        )
        for k, v in episodes_stats.items():
            metrics.setdefault(k, []).extend(v)

        episode_table = dump_episodes(
            episodes=episodes,
            episodes_stats=episodes_stats,
            exp_dir=EXP_DIR,
            tokenizer=tokenizer,
            iteration=iteration,
        )

        #########################################################
        # Training
        #########################################################

        # Prepare training batch
        model_inputs = prepare_model_inputs(query_token_ids=episodes["all_query_token_ids"],
                                            response_token_ids=episodes["all_response_token_ids"],
                                            advantages=episodes["all_advantages"],
                                            device="cuda")

        # Calculate losses and update model
        policy_model.train()
        reference_model.module.cuda()
        reference_model.eval()

        total_response_len = (model_inputs["labels"] != -100).sum().item()

        for i in trange(0, args.episodes_per_iteration, args.per_device_batch_size, desc="Gradient Accumulation"):
            batch = {k: v[i:i + args.per_device_batch_size] for k, v in model_inputs.items()}

            # Compute policy gradient loss
            loss, loss_metrics = compute_pg_loss(
                policy_model=policy_model,
                reference_model=reference_model,
                batch=batch,
                total_response_len=total_response_len,
                temperature=args.temperature,
                kl_coefficient=args.kl_coefficient,
            )

            # Track metrics
            grad_norm = policy_model.get_global_grad_norm()
            if grad_norm is not None:
                grad_norm = grad_norm.item()

            metrics.setdefault("grad_norm", []).append(grad_norm)
            metrics.setdefault("loss", []).append(loss.item())

            for k, v in loss_metrics.items():
                metrics.setdefault(k, []).append(v.item() if isinstance(v, torch.Tensor) else v)

            # Backpropagation and optimization step
            policy_model.backward(loss, scale_wrt_gas=False)

            # Free memory
            del loss, loss_metrics
            if policy_model.is_gradient_accumulation_boundary():
                reference_model.module.cpu()

            policy_model.step()

        #########################################################
        # Update inference engine weights
        #########################################################

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        inference_engine.wake_up()
        load_model_into_vllm(policy_model, inference_engine)

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        #########################################################
        # Log metrics
        #########################################################

        train_metrics = {k: np.mean(v) for k, v in metrics.items() if None not in v}
        train_metrics["learning_rate"] = policy_model.get_lr()[0]
        logs = {
            "iteration": iteration,
            f"episodes/iter_{iteration:06d}": episode_table,
            **{
                f"train/{k}": v for k, v in train_metrics.items()
            },
        }
        if eval_stats is not None:
            eval_metrics = {k: np.mean(v) for k, v in eval_stats.items() if None not in v}
            logs.update({f"eval/{k}": v for k, v in eval_metrics.items()})
        wandb.log(logs)

        selected_keys = [
            "train/kl_penalty",
            "train/rewards",
            "train/reward_metrics/format_reward",
            "train/reward_metrics/equation_reward",
            "eval/rewards",
            "eval/reward_metrics/format_reward",
            "eval/reward_metrics/equation_reward",
        ]
        selected_metrics = {k: logs[k] for k in selected_keys if k in logs}
        print(f"KEY METRICS: {selected_metrics}")

        if iteration % args.checkpoint_interval == 0 and iteration != 0:
            policy_model.module.save_pretrained(str(EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}" / "hf_model"))
            policy_model.save_checkpoint(str(EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}" / "deepspeed"))
