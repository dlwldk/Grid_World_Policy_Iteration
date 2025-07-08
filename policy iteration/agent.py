import numpy as np
from visualize_train import draw_value_image, draw_policy_image

# left, right, up, down
ACTIONS = [np.array([0, -1]),
           np.array([0, 1]),
           np.array([-1, 0]),
           np.array([1, 0])]


class AGENT:
    def __init__(self, env, is_upload=False):

        self.ACTIONS = ACTIONS
        self.env = env
        HEIGHT, WIDTH = env.size()
        self.state = [0,0]

        if is_upload:
            dp_results = np.load('./result/dp.npz')
            self.values = dp_results['V']
            self.policy = dp_results['PI']
        else:
            self.values = np.zeros((HEIGHT, WIDTH))
            self.policy = np.zeros((HEIGHT, WIDTH,len(self.ACTIONS)))+1./len(self.ACTIONS)




    def policy_evaluation(self, iter, env, policy, discount=1.0):
        HEIGHT, WIDTH = env.size()
        new_state_values = np.zeros((HEIGHT, WIDTH))
        iteration = 0
        
        # 주어진 정책에 대한 value function을 반복적으로 계산하여 수렴할 때까지 진행
        while True:
            delta = 0  # value function의 최대 변화량
            # 모든 state에 대해 반복
            for i in range(HEIGHT): # HEIGHT만큼 반복
                for j in range(WIDTH): # WIDTH만큼 반복
                    state = [i, j]  # 현재 state
                    if env.is_terminal(state):  # 목표 상태인 경우 건너뛰기
                        continue
                        
                    temp = new_state_values[i,j]  # 현재 상태의 value 값 임시 저장
                    v = 0  # 현재 상태의 새로운 value 값
                    
                    for a, action in enumerate(self.ACTIONS): # 모든 가능한 행동에 대해 반복
                        next_state, reward = env.interaction(state, action) # environment와 상호작용하여 다음 state와 reward 얻기
                        # Bellman equation을 사용하여 value function 업데이트
                        v += policy[i,j,a] * (reward + discount * new_state_values[next_state[0], next_state[1]])
                    
                    # value function 최대 변화량 업데이트(delta와 temp-v값 중 최대값 선택)
                    delta = max(delta, abs(temp - v))
                    new_state_values[i,j] = v
            
            iteration += 1  # 반복 횟수 증가
            # 수렴 조건 (delta(value function의 변화값)가 충분히 작을 때(θ보다 작을 때))
            if delta < 1e-4:  # θ보다 작은 경우 break
                break

        draw_value_image(iter, np.round(new_state_values, decimals=2), env=env)
        return new_state_values, iteration

    def policy_improvement(self, iter, env, state_values, old_policy, discount=1.0):
        HEIGHT, WIDTH = env.size()
        policy = old_policy.copy()

        policy_stable = True  # policy 값 안정여부 flag
        
        # 모든 state에 대해 반복
        for i in range(HEIGHT): # HEIGHT만큼 반복
            for j in range(WIDTH): # WIDTH만큼 반복
                state = [i, j] # 현재 state
                if env.is_terminal(state):  # 목표 상태인 경우 건너뛰기
                    continue
                
                temp = policy[i,j,:].copy() # 현재 상태의 policy 값 임시 저장
                
                # 각 action에 대한 value 계산
                action_values = []
                for a, action in enumerate(self.ACTIONS):
                    next_state, reward = env.interaction(state, action) # environment와 상호작용하여 다음 state와 reward 얻기
                    # Bellman equation을 사용하여 action value 계산
                    action_value = reward + discount * state_values[next_state[0], next_state[1]]
                    action_values.append(action_value) # action_value 값 추가
                
                # 최적의 action 선택 (최대 가치를 가진 행동)
                best_action = np.argmax(action_values)
                
                # 정책 결정 후 업데이트
                policy[i,j,:] = 0
                policy[i,j,best_action] = 1
                
                # 정책이 변경되었는지 확인
                if not np.array_equal(temp, policy[i,j,:]):
                    policy_stable = False

        print('policy stable {}:'.format(policy_stable))
        draw_policy_image(iter, np.round(policy, decimals=2), env=env)
        
        return policy, policy_stable

    def policy_iteration(self):
        iter = 1
        while (True):
            self.values, iteration = self.policy_evaluation(iter, env=self.env, policy=self.policy)
            self.policy, policy_stable = self.policy_improvement(iter, env=self.env, state_values=self.values,
                                                       old_policy=self.policy, discount=1.0)
            iter += 1
            if policy_stable == True:
                break
        np.savez('./result/dp.npz', V=self.values, PI=self.policy)
        return self.values, self.policy



    def get_action(self, state):
        i,j = state
        return np.random.choice(len(ACTIONS), 1, p=self.policy[i,j,:].tolist()).item()


    def get_state(self):
        return self.state

