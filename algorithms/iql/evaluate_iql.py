"""评估训练好的IQL模型

功能：
1. 加载检查点
2. 在数据集上计算离线指标：
   - 平均Q值
   - 平均V值
   - Q值预测的行为策略回报
   - 贪心策略（max_a Q(s,a)）的期望回报
3. 保存评估结果
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from algorithms.iql.dataset import ReadyDataset
from algorithms.iql.models import QNetwork, VNetwork, GaussianPolicy


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(ckpt_path: str, state_dim: int, action_dim: int, hidden: list):
    """加载检查点并返回Q/V/Policy网络"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    q_net = QNetwork(state_dim, action_dim, hidden)
    v_net = VNetwork(state_dim, hidden)
    pi_net = GaussianPolicy(state_dim, action_dim, hidden)
    
    q_net.load_state_dict(ckpt['q_state'])
    v_net.load_state_dict(ckpt['v_state'])
    pi_net.load_state_dict(ckpt['pi_state'])
    
    q_net.eval()
    v_net.eval()
    pi_net.eval()
    
    return q_net, v_net, pi_net


def evaluate_offline(dataset: ReadyDataset, q_net, v_net, pi_net, gamma=0.99):
    """在数据集上计算离线评估指标"""
    trans_df = dataset.to_transitions()
    
    # 转换为张量
    states = torch.FloatTensor(np.stack(trans_df['s'].values))
    actions = torch.FloatTensor(trans_df['a'].values).unsqueeze(1)
    rewards = torch.FloatTensor(trans_df['r'].values)
    next_states = torch.FloatTensor(np.stack(trans_df['s_next'].values))
    dones = torch.FloatTensor(trans_df['done'].values)
    
    with torch.no_grad():
        # Q值评估
        q_values = q_net(states, actions)
        mean_q = q_values.mean().item()
        std_q = q_values.std().item()
        
        # V值评估
        v_values = v_net(states)
        mean_v = v_values.mean().item()
        std_v = v_values.std().item()
        
        # 策略分布（行为克隆质量）
        pi_mean, pi_logstd = pi_net(states)
        pi_std = torch.exp(pi_logstd)
        # 计算策略与行为动作的NLL
        log_prob = -0.5 * ((actions - pi_mean) / pi_std) ** 2 - pi_logstd - 0.5 * np.log(2 * np.pi)
        mean_log_prob = log_prob.mean().item()
        
        # 贪心策略评估（用Q网络做动作选择）
        # 在动作空间[-1, 1]上采样，找最大Q值
        n_samples = 20
        action_samples = torch.linspace(-1, 1, n_samples).unsqueeze(0).unsqueeze(2)  # [1, n_samples, 1]
        states_expanded = states.unsqueeze(1).expand(-1, n_samples, -1)  # [N, n_samples, state_dim]
        
        # 计算所有采样动作的Q值
        q_samples = []
        for i in range(0, len(states), 256):  # 批处理避免内存溢出
            batch_states = states_expanded[i:i+256].reshape(-1, states.shape[1])
            batch_actions = action_samples.expand(min(256, len(states)-i), -1, -1).reshape(-1, 1)
            batch_q = q_net(batch_states, batch_actions).reshape(-1, n_samples)
            q_samples.append(batch_q)
        
        q_samples = torch.cat(q_samples, dim=0)  # [N, n_samples]
        max_q_values = q_samples.max(dim=1)[0]
        greedy_actions = action_samples.squeeze().unsqueeze(0).expand(len(states), -1)[torch.arange(len(states)), q_samples.argmax(dim=1)]
        
        mean_greedy_q = max_q_values.mean().item()
        
        # 计算MC回报（实际数据的累积回报）
        mc_returns = []
        current_return = 0
        discount = 1.0
        for i in range(len(rewards)-1, -1, -1):
            current_return = rewards[i] + gamma * current_return * (1 - dones[i])
            mc_returns.insert(0, current_return.item())
            if dones[i] > 0.5:
                discount = 1.0
        
        mean_mc_return = np.mean(mc_returns)
    
    results = {
        'num_transitions': len(trans_df),
        'q_stats': {'mean': mean_q, 'std': std_q},
        'v_stats': {'mean': mean_v, 'std': std_v},
        'greedy_q': mean_greedy_q,
        'policy_log_prob': mean_log_prob,
        'mc_return': mean_mc_return,
        'mean_reward': rewards.mean().item(),
    }
    
    return results


def print_evaluation(results: dict):
    """打印评估结果"""
    print("=" * 70)
    print("IQL 离线评估结果")
    print("=" * 70)
    print(f"\n数据集信息:")
    print(f"  转换数量: {results['num_transitions']}")
    print(f"  平均奖励: {results['mean_reward']:.4f}")
    print(f"  MC回报: {results['mc_return']:.4f}")
    
    print(f"\n价值函数评估:")
    print(f"  Q值 - 均值: {results['q_stats']['mean']:.4f}, 标准差: {results['q_stats']['std']:.4f}")
    print(f"  V值 - 均值: {results['v_stats']['mean']:.4f}, 标准差: {results['v_stats']['std']:.4f}")
    
    print(f"\n策略评估:")
    print(f"  贪心策略Q值: {results['greedy_q']:.4f}")
    print(f"  策略对数似然: {results['policy_log_prob']:.4f}")
    
    print(f"\n性能指标:")
    improvement = ((results['greedy_q'] - results['q_stats']['mean']) / abs(results['q_stats']['mean'])) * 100
    print(f"  贪心策略相对行为策略改进: {improvement:+.1f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="评估IQL模型")
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    parser.add_argument("--config", type=str, required=True, help="训练配置文件")
    parser.add_argument("--output", type=str, default=None, help="结果保存路径（可选）")
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 加载数据
    print(f"加载数据: {cfg['data']['path']}")
    import pandas as pd
    df = pd.read_csv(cfg['data']['path'])
    dataset = ReadyDataset(
        df=df,
        state_cols=cfg['data']['state_cols']
    )
    
    state_dim = len(cfg['data']['state_cols'])
    action_dim = 1
    
    # 加载模型
    print(f"加载检查点: {args.checkpoint}")
    q_net, v_net, pi_net = load_checkpoint(
        args.checkpoint,
        state_dim,
        action_dim,
        cfg['model']['hidden']
    )
    
    # 评估
    print("\n开始评估...")
    results = evaluate_offline(dataset, q_net, v_net, pi_net, cfg['model']['gamma'])
    
    # 打印结果
    print_evaluation(results)
    
    # 保存结果
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
