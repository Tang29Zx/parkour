"""Run frozen-policy reward audits for the Go2 five-box task."""

import json
from collections import defaultdict

import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.
from isaacgym import gymtorch
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args
from legged_gym.utils.task_registry import task_registry


def _scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().item())
    return float(value)


class EpisodeAudit:
    """Accumulate reset-batch metrics with the correct episode weights."""

    def __init__(self):
        self.episode_count = 0.0
        self.weighted = defaultdict(float)
        self.weights = defaultdict(float)

    def add(self, episode):
        count = _scalar(episode.get("num_terminated", 0.0))
        if count <= 0.0:
            return
        self.episode_count += count
        for key, value in episode.items():
            if key.endswith("_episode_count"):
                continue
            weight = count
            if key.startswith("raw/") and key.endswith("_mean_return"):
                result = key[len("raw/") : -len("_mean_return")]
                weight = _scalar(
                    episode.get(f"raw/{result}_episode_count", 0.0)
                )
            if weight <= 0.0:
                continue
            self.weighted[key] += _scalar(value) * weight
            self.weights[key] += weight

    def summary(self):
        result = {"episode_count": self.episode_count}
        for key, total in self.weighted.items():
            result[key] = total / max(self.weights[key], 1.0)
        return result


def _actions_for_mode(mode, actor_critic, obs, env, episode_age, args):
    if mode in ("zero", "delayed_policy", "late_failure"):
        actions = torch.zeros(
            env.num_envs, env.num_actions, device=env.device
        )
        if mode == "delayed_policy":
            active = episode_age >= int(args.delay_s / env.dt)
            if active.any():
                policy_actions = actor_critic.act_inference(obs)
                actions[active] = policy_actions[active]
        return actions
    return actor_critic.act_inference(obs)


@torch.no_grad()
def run_mode(mode, env, actor_critic, args):
    obs, _ = env.reset()
    actor_critic.reset(
        torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    )
    episode_age = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    audit = EpisodeAudit()

    while audit.episode_count < args.audit_episodes:
        if mode == "stand":
            env.commands.zero_()
            env.compute_observations()
            obs = env.get_observations()
        if mode == "early_failure":
            env.root_states[:, 2] = env.env_origins[:, 2] + 0.05
            env.gym.set_actor_root_state_tensor(
                env.sim, gymtorch.unwrap_tensor(env.all_root_states)
            )
        elif mode == "late_failure":
            fail_now = episode_age >= int(args.late_failure_s / env.dt)
            if fail_now.any():
                env.root_states[fail_now, 2] = (
                    env.env_origins[fail_now, 2] + 0.05
                )
                env.gym.set_actor_root_state_tensor(
                    env.sim, gymtorch.unwrap_tensor(env.all_root_states)
                )

        actions = _actions_for_mode(
            mode, actor_critic, obs.detach(), env, episode_age, args
        )
        obs, _, _, dones, infos = env.step(actions.detach())
        episode_age += 1
        # ``env.extras`` is persistent, so an old episode dictionary can be
        # returned again on non-terminal steps. Count it only with fresh dones.
        if dones.any() and "episode" in infos and infos["episode"]:
            audit.add(infos["episode"])
        if dones.any():
            actor_critic.reset(dones)
            episode_age[dones] = 0
    return audit.summary()


def _reward_audit_warnings(results):
    normal = results.get("policy", {})
    success_return = normal.get("raw/success_mean_return")
    comparisons = (
        ("zero", "raw/incomplete_mean_return", "standing/zero-action"),
        ("early_failure", "raw/early_failure_mean_return", "early failure"),
        ("late_failure", "raw/fall_failure_mean_return", "late failure"),
    )
    warnings = []
    if "policy" not in results:
        print(
            "Reward ordering check skipped: include the policy mode together "
            "with one or more baseline modes."
        )
        return
    if success_return is None:
        warnings.append("The policy produced no successful episode to audit.")
    else:
        for mode, key, label in comparisons:
            baseline = results.get(mode, {}).get(key)
            if baseline is not None and success_return <= baseline:
                warnings.append(
                    f"Successful return {success_return:.3f} is not greater "
                    f"than {label} return {baseline:.3f}."
                )
    if warnings:
        print("\033[1;31mREWARD LOOPHOLE WARNING")
        for warning in warnings:
            print(f"  - {warning}")
        print("\033[0m")
    else:
        print("Reward ordering checks passed for all available baselines.")


def audit(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    if args.num_envs is None:
        env_cfg.env.num_envs = 256
    train_cfg.runner.resume = True
    train_cfg.runner.ckpt_manipulator = None
    env, env_cfg = task_registry.make_env(
        args.task, args=args, env_cfg=env_cfg
    )
    runner, _ = task_registry.make_alg_runner(
        env=env,
        args=args,
        train_cfg=train_cfg,
        env_cfg=env_cfg,
        log_root=None,
        save_cfg=False,
    )
    actor_critic = runner.alg.actor_critic
    actor_critic.eval()

    modes = [mode.strip() for mode in args.audit_modes.split(",") if mode.strip()]
    supported = {
        "policy",
        "stand",
        "zero",
        "delayed_policy",
        "early_failure",
        "late_failure",
    }
    unknown = sorted(set(modes) - supported)
    if unknown:
        raise ValueError(f"Unknown audit modes: {unknown}")

    results = {}
    for mode in modes:
        print(f"Running frozen audit mode: {mode}")
        results[mode] = run_mode(mode, env, actor_critic, args)
        print(json.dumps(results[mode], indent=2, sort_keys=True))
    _reward_audit_warnings(results)


if __name__ == "__main__":
    audit(
        get_args(
            [
                {
                    "name": "--audit_episodes",
                    "type": int,
                    "default": 512,
                    "help": "Minimum completed episodes per frozen mode.",
                },
                {
                    "name": "--audit_modes",
                    "type": str,
                    "default": "policy,stand,zero,delayed_policy,early_failure,late_failure",
                    "help": "Comma-separated frozen-policy audit modes.",
                },
                {
                    "name": "--delay_s",
                    "type": float,
                    "default": 5.0,
                    "help": "Zero-action delay before policy execution.",
                },
                {
                    "name": "--late_failure_s",
                    "type": float,
                    "default": 40.0,
                    "help": "Delay before forcing the late-failure baseline.",
                },
            ]
        )
    )
