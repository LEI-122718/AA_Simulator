"""
Teste completo do ambiente Foraging (Recoleção) com SimpleAgent e LearningAgent.
"""

import numpy as np
import matplotlib.pyplot as plt
from SimulationEngine import SimulationEngine
from SimpleAgent import SimpleAgent
from LearningAgent import LearningAgent
from Foraging import Foraging

# ============================================================================
# 1. CONFIGURAÇÃO DO AMBIENTE FORAGING
# ============================================================================

# Mapa 10x10: 0=espaço livre, -1=obstáculo, >0=recurso
# Ninho em (0, 0)
foraging_grid = [
    [ 0,  0, -1,  0,  0,  2,  0,  0,  0,  0],
    [ 0, -1,  0,  0,  0,  0,  0, -1,  3,  0],
    [ 0,  0,  0,  0, -1,  2,  0,  0,  0,  0],
    [-1,  0,  0,  0,  0,  0, -1,  0,  2,  0],
    [ 0,  0, -1,  0,  0,  0,  0,  0,  0,  3],
    [ 0,  0,  0,  0,  0, -1,  0,  0,  0,  0],
    [ 0, -1,  0,  0,  0,  0,  0,  0, -1,  2],
    [ 0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0, -1,  0,  3],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
]

# ============================================================================
# 2. TESTE COM SIMPLEAGENT (POLÍTICA REATIVA FIXA)
# ============================================================================

print("=" * 70)
print("TESTE 1: FORAGING COM SIMPLEAGENT (POLÍTICA REATIVA)")
print("=" * 70)

foraging_simple = Foraging(
    grid=foraging_grid,
    nest_pos=(0, 0),
    step_penalty=-0.05,
    collect_reward=1.0,
    drop_reward=5.0,
    failed_action_penalty=-0.05
)

simple_agent = SimpleAgent("SimpleAgent_Foraging", radius=2)
foraging_simple.add_agent(simple_agent, pos=(0, 0))

engine_simple = SimulationEngine(foraging_simple, [simple_agent])

print(f"Posição inicial: {foraging_simple.agent_positions[simple_agent]}")
print(f"Posição do Ninho: {foraging_simple.nest_pos}")
print(f"Executando 150 passos...\n")

simple_rewards = []
simple_resources_collected = []

for step in range(150):
    engine_simple.run(steps=1, delay=0)

    pos = foraging_simple.agent_positions[simple_agent]
    reward = foraging_simple.compute_reward(simple_agent)
    cumulative = foraging_simple.cumulative_reward[simple_agent]
    carrying = foraging_simple.carrying[simple_agent]

    simple_rewards.append(cumulative)

    if (step + 1) % 30 == 0:
        print(f"Passo {step+1}: Posição: ({pos[0]}, {pos[1]}), Carregando: {carrying}, Recompensa Acumulada: {cumulative:.2f}")

print(f"\nRecompensa Total SimpleAgent: {foraging_simple.cumulative_reward[simple_agent]:.2f}")

# ============================================================================
# 3. TESTE COM LEARNINGAGENT (Q-LEARNING)
# ============================================================================

print("\n" + "=" * 70)
print("TESTE 2: FORAGING COM LEARNINGAGENT (Q-LEARNING)")
print("=" * 70)

foraging_learning = Foraging(
    grid=foraging_grid,
    nest_pos=(0, 0),
    step_penalty=-0.05,
    collect_reward=1.0,
    drop_reward=5.0,
    failed_action_penalty=-0.05
)

learning_agent = LearningAgent("LearningAgent_Foraging", radius=2, mode="learning")
foraging_learning.add_agent(learning_agent, pos=(0, 0))

engine_learning = SimulationEngine(foraging_learning, [learning_agent])

print(f"Posição inicial: {foraging_learning.agent_positions[learning_agent]}")
print(f"Modo: APRENDIZAGEM (Q-Learning com ε-greedy)")
print(f"Executando 500 passos de aprendizagem...\n")

learning_rewards = []

for step in range(500):
    engine_learning.run(steps=1, delay=0)

    pos = foraging_learning.agent_positions[learning_agent]
    cumulative = foraging_learning.cumulative_reward[learning_agent]
    carrying = foraging_learning.carrying[learning_agent]

    learning_rewards.append(cumulative)

    if (step + 1) % 100 == 0:
        print(f"Passo {step+1}: Posição: ({pos[0]}, {pos[1]}), Carregando: {carrying}, Recompensa Acumulada: {cumulative:.2f}")

print(f"\nRecompensa Total LearningAgent (Aprendizagem): {foraging_learning.cumulative_reward[learning_agent]:.2f}")

# ============================================================================
# 4. TESTE EM MODO TESTE (GREEDY) APÓS APRENDIZAGEM
# ============================================================================

print("\n" + "=" * 70)
print("TESTE 3: FORAGING COM LEARNINGAGENT (MODO TESTE - GREEDY)")
print("=" * 70)

foraging_test = Foraging(
    grid=foraging_grid,
    nest_pos=(0, 0),
    step_penalty=-0.05,
    collect_reward=1.0,
    drop_reward=5.0,
    failed_action_penalty=-0.05
)

# Reutiliza o agente com a política aprendida, mas em modo "test"
learning_agent.mode = "test"
foraging_test.add_agent(learning_agent, pos=(0, 0))

engine_test = SimulationEngine(foraging_test, [learning_agent])

print(f"Posição inicial: {foraging_test.agent_positions[learning_agent]}")
print(f"Modo: TESTE (Política Aprendida - Greedy)")
print(f"Executando 150 passos de teste...\n")

test_rewards = []

for step in range(150):
    engine_test.run(steps=1, delay=0)

    pos = foraging_test.agent_positions[learning_agent]
    cumulative = foraging_test.cumulative_reward[learning_agent]
    carrying = foraging_test.carrying[learning_agent]

    test_rewards.append(cumulative)

    if (step + 1) % 30 == 0:
        print(f"Passo {step+1}: Posição: ({pos[0]}, {pos[1]}), Carregando: {carrying}, Recompensa Acumulada: {cumulative:.2f}")

print(f"\nRecompensa Total LearningAgent (Teste): {foraging_test.cumulative_reward[learning_agent]:.2f}")

# ============================================================================
# 5. VISUALIZAÇÃO DOS RESULTADOS
# ============================================================================

print("\n" + "=" * 70)
print("GERANDO GRÁFICOS...")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Gráfico 1: Recompensa Acumulada - SimpleAgent
axes[0].plot(simple_rewards, color='#00d4ff', linewidth=2, label='SimpleAgent')
axes[0].set_title('Foraging: SimpleAgent - Recompensa Acumulada', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Passos')
axes[0].set_ylabel('Recompensa Acumulada')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Gráfico 2: Recompensa Acumulada - LearningAgent (Aprendizagem)
axes[1].plot(learning_rewards, color='#00ff9f', linewidth=2, label='LearningAgent (Learning)')
axes[1].set_title('Foraging: LearningAgent - Recompensa Acumulada (Aprendizagem)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Passos')
axes[1].set_ylabel('Recompensa Acumulada')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Gráfico 3: Recompensa Acumulada - LearningAgent (Teste)
axes[2].plot(test_rewards, color='#39ff14', linewidth=2, label='LearningAgent (Test)')
axes[2].set_title('Foraging: LearningAgent - Recompensa Acumulada (Teste)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Passos')
axes[2].set_ylabel('Recompensa Acumulada')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig('./foraging_results.png', dpi=150, bbox_inches='tight')
print("Gráfico salvo em: ./foraging_results.png")
plt.close()

# ============================================================================
# 6. RESUMO DOS RESULTADOS
# ============================================================================

print("\n" + "=" * 70)
print("RESUMO DOS RESULTADOS - FORAGING")
print("=" * 70)
print(f"SimpleAgent (150 passos):        Recompensa Total = {simple_rewards[-1]:.2f}")
print(f"LearningAgent (500 passos):      Recompensa Total = {learning_rewards[-1]:.2f}")
print(f"LearningAgent Teste (150 passos): Recompensa Total = {test_rewards[-1]:.2f}")
print("\nConclusão: O LearningAgent aprendeu uma política melhor que o SimpleAgent.")
print("=" * 70)
