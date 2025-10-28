"""
环境兼容性包装器
解决 gym-sepsis 使用旧 gym 的问题
"""
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='gym')
import numpy as np

try:
    import gymnasium as gym_new
    USING_GYMNASIUM = True
except ImportError:
    import gym as gym_new
    USING_GYMNASIUM = False

import gym as old_gym  # gym-sepsis 使用旧版 gym
import gym_sepsis


def make_sepsis_env():
    """
    创建 Gym-Sepsis 环境，自动处理 gym/gymnasium 兼容性
    
    Returns:
        env: Sepsis 环境实例
    """
    # gym-sepsis 注册在旧 gym 中，版本是 v0
    env = old_gym.make('sepsis-v0')
    
    print(f"✅ Sepsis environment created (sepsis-v0)")
    print(f"   State space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.n} actions")
    
    return env


def get_feature_names():
    """
    返回 46 个特征的名称（按顺序）
    来源：项目文档
    """
    return [
        'albumin', 'anion_gap', 'band_neutrophils', 'bicarbonate', 'bilirubin',
        'bun', 'chloride', 'creatinine', 'dbp', 'glucose_1', 'glucose_2',
        'heart_rate', 'hematocrit', 'hemoglobin', 'inr', 'lactate',
        'map', 'paco2', 'platelet_count', 'potassium', 'pt', 'ptt',
        'respiratory_rate', 'sodium', 'spo2', 'sbp', 'temp_c', 'wbc',
        'age', 'gender', 'race_white', 'race_black', 'race_hispanic', 'race_other',
        'height', 'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa',
        'qsofa_sbp', 'qsofa_gcs', 'qsofa_rr', 'elixhauser', 'blood_culture_positive'
    ]


def print_state_info(state):
    """
    打印状态中的关键临床特征（用于调试）
    
    Args:
        state: 46维状态向量
    """
    feature_names = get_feature_names()
    
    # 关键特征及其索引
    key_features = {
        'lactate': 15,
        'sbp': 25,
        'heart_rate': 11,
        'map': 16,
        'sofa': 37,
        'wbc': 27,
        'respiratory_rate': 22
    }
    
    print("\n🩺 Key Clinical Features:")
    for name, idx in key_features.items():
        value = state[idx] if idx < len(state) else 'N/A'
        print(f"   {name:20s}: {value:.2f}" if isinstance(value, (int, float)) else f"   {name:20s}: {value}")


def test_environment():
    """
    测试环境是否正常工作
    """
    env = make_sepsis_env()
    
    # 运行一个简单的 episode
    # 处理不同版本 Gym 的返回值
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
    
    print(f"\n📊 Initial observation:")
    print(f"   Shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}")
    print(f"   Expected: (46,)")
    
    # 打印初始状态的关键特征
    print_state_info(obs)
    
    done = False
    step_count = 0
    total_reward = 0
    
    print("\n🏃 Running test episode...")
    while not done and step_count < 100:  # 限制最大步数
        action = env.action_space.sample()  # 随机动作
        
        # 处理不同版本 Gym 的 step 返回值
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        
        total_reward += reward
        step_count += 1
        
        # 每 10 步打印一次进度
        if step_count % 10 == 0:
            print(f"   Step {step_count}: reward={reward}, done={done}")
    
    print(f"\n✅ Test episode completed:")
    print(f"   Steps: {step_count}")
    print(f"   Total reward: {total_reward}")
    print(f"   Outcome: {'Survived ✅' if total_reward > 0 else 'Died ❌'}")
    
    # 打印最终状态
    print_state_info(obs)
    
    env.close()
    return True


if __name__ == "__main__":
    print("🧪 Testing Gym-Sepsis environment...\n")
    test_environment()