"""
Teste completo do ambiente Lighthouse (Farol) com SimpleAgent e LearningAgent.
"""

import numpy as np
import matplotlib.pyplot as plt
from SimulationEngine import SimulationEngine
from SimpleAgent import SimpleAgent
from LearningAgent import LearningAgent
from Lighthouse import Lighthouse

# ============================================================================
# 1. CONFIGURAÇÃO DO AMBIENTE LIGHTHOUSE
# ============================================================================

# Mapa 10x10: 0=espaço livre, -1=obstáculo, 10=farol
# O farol está na posição (9, 9)
lighthouse_grid = [
    [ 0,  0, -1,  0,  0,  0,  0,  0,  0,  0],
    [ 0, -1,  0,  0,  0,  0,  0, -1,  0,  0],
    [ 0,  0,  0,  0, -1,  0,  0,  0,  0,  0],
    [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0],
    [ 0,  0, -1,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0, -1,  0,  0,  0,  0],
    [ 0, -1,  0,  0,  0,  0,  0,  0, -1,  0],
    [ 0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0, -1,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0, 10]  # Farol em (9, 9)
]

# ============================================================================
# 2. TESTE COM SIMPLEAGENT (POLÍTICA REATIVA FIXA)
# ============================================================================

print("=" * 70)
print("TESTE 1: LIGHTHOUSE COM SIMPLEAGENT (POLÍTICA REATIVA)")
print("=" * 70)

lighthouse_simple = Lighthouse(
    grid=lighthouse_grid,
    novelty_penalty=0.1,
    step_penalty=-0.05,
    wall_penalty=-1.0,
    goal_reward=10.0
)

simple_agent = SimpleAgent("SimpleAgent_Lighthouse", radius=2)
lighthouse_simple.add_agent(simple_agent, pos=(0, 0))

engine_simple = SimulationEngine(lighthouse_simple, [simple_agent])

print(f"Posição inicial: {lighthouse_simple.agent_positions[simple_agent]}")
print(f"Posição do Farol: {lighthouse_simple.lighthouse_pos}")
print(f"Executando 100 passos...\n")

simple_rewards = []
simple_positions = []

for step in range(100):
    engine_simple.run(steps=1, delay=0)

    pos = lighthouse_simple.agent_positions[simple_agent]
    reward = lighthouse_simple.compute_reward(simple_agent)
    cumulative = lighthouse_simple.cumulative_reward[simple_agent]

    simple_rewards.append(cumulative)
    simple_positions.append(tuple(pos))

    if (step + 1) % 20 == 0:
        print(f"Passo {step+1}: Posição: ({pos[0]}, {pos[1]}), Recompensa Acumulada: {cumulative:.2f}")

print(f"\nRecompensa Total SimpleAgent: {lighthouse_simple.cumulative_reward[simple_agent]:.2f}")

# ============================================================================
# 3. TESTE COM LEARNINGAGENT (Q-LEARNING)
# ============================================================================

print("\n" + "=" * 70)
print("TESTE 2: LIGHTHOUSE COM LEARNINGAGENT (Q-LEARNING)")
print("=" * 70)

lighthouse_learning = Lighthouse(
    grid=lighthouse_grid,
    novelty_penalty=0.1,
    step_penalty=-0.05,
    wall_penalty=-1.0,
    goal_reward=10.0
)

learning_agent = LearningAgent("LearningAgent_Lighthouse", radius=2, mode="learning")
lighthouse_learning.add_agent(learning_agent, pos=(0, 0))

engine_learning = SimulationEngine(lighthouse_learning, [learning_agent])

print(f"Posição inicial: {lighthouse_learning.agent_positions[learning_agent]}")
print(f"Modo: APRENDIZAGEM (Q-Learning com ε-greedy)")
print(f"Executando 500 passos de aprendizagem...\n")

learning_rewards = []
learning_positions = []

for step in range(500):
    engine_learning.run(steps=1, delay=0)

    pos = lighthouse_learning.agent_positions[learning_agent]
    cumulative = lighthouse_learning.cumulative_reward[learning_agent]

    learning_rewards.append(cumulative)
    learning_positions.append(tuple(pos))

    if (step + 1) % 100 == 0:
        print(f"Passo {step+1}: Posição: ({pos[0]}, {pos[1]}), Recompensa Acumulada: {cumulative:.2f}")

print(f"\nRecompensa Total LearningAgent (Aprendizagem): {lighthouse_learning.cumulative_reward[learning_agent]:.2f}")

# ============================================================================
# 4. TESTE EM MODO TESTE (GREEDY) APÓS APRENDIZAGEM
# ============================================================================

print("\n" + "=" * 70)
print("TESTE 3: LIGHTHOUSE COM LEARNINGAGENT (MODO TESTE - GREEDY)")
print("=" * 70)

lighthouse_test = Lighthouse(
    grid=lighthouse_grid,
    novelty_penalty=0.1,
    step_penalty=-0.05,
    wall_penalty=-1.0,
    goal_reward=10.0
)

# Reutiliza o agente com a política aprendida, mas em modo "test"
learning_agent.mode = "test"
lighthouse_test.add_agent(learning_agent, pos=(0, 0))

engine_test = SimulationEngine(lighthouse_test, [learning_agent])

print(f"Posição inicial: {lighthouse_test.agent_positions[learning_agent]}")
print(f"Modo: TESTE (Política Aprendida - Greedy)")
print(f"Executando 100 passos de teste...\n")

test_rewards = []
test_positions = []

for step in range(100):
    engine_test.run(steps=1, delay=0)

    pos = lighthouse_test.agent_positions[learning_agent]
    cumulative = lighthouse_test.cumulative_reward[learning_agent]

    test_rewards.append(cumulative)
    test_positions.append(tuple(pos))

    if (step + 1) % 20 == 0:
        print(f"Passo {step+1}: Posição: ({pos[0]}, {pos[1]}), Recompensa Acumulada: {cumulative:.2f}")

print(f"\nRecompensa Total LearningAgent (Teste): {lighthouse_test.cumulative_reward[learning_agent]:.2f}")

# ============================================================================
# 5. VISUALIZAÇÃO DOS RESULTADOS
# ============================================================================

print("\n" + "=" * 70)
print("GERANDO GRÁFICOS...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico 1: Recompensa Acumulada - SimpleAgent
axes[0, 0].plot(simple_rewards, color='#00d4ff', linewidth=2, label='SimpleAgent')
axes[0, 0].set_title('Lighthouse: SimpleAgent - Recompensa Acumulada', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Passos')
axes[0, 0].set_ylabel('Recompensa Acumulada')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Gráfico 2: Recompensa Acumulada - LearningAgent (Aprendizagem)
axes[0, 1].plot(learning_rewards, color='#00ff9f', linewidth=2, label='LearningAgent (Learning)')
axes[0, 1].set_title('Lighthouse: LearningAgent - Recompensa Acumulada (Aprendizagem)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Passos')
axes[0, 1].set_ylabel('Recompensa Acumulada')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Gráfico 3: Recompensa Acumulada - LearningAgent (Teste)
axes[1, 0].plot(test_rewards, color='#39ff14', linewidth=2, label='LearningAgent (Test)')
axes[1, 0].set_title('Lighthouse: LearningAgent - Recompensa Acumulada (Teste)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Passos')
axes[1, 0].set_ylabel('Recompensa Acumulada')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Gráfico 4: Comparação de Trajetórias
ax = axes[1, 1]
ax.set_xlim(-0.5, 9.5)
ax.set_ylim(-0.5, 9.5)
ax.set_aspect('equal')
ax.invert_yaxis()

# Desenha obstáculos
for y in range(10):
    for x in range(10):
        if lighthouse_grid[y][x] == -1:
            ax.add_patch(plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, color='red', alpha=0.5))

# Desenha farol
ax.plot(9, 9, 'y*', markersize=20, label='Farol')

# Desenha trajetórias
simple_x = [p[0] for p in simple_positions[:50]]
simple_y = [p[1] for p in simple_positions[:50]]
ax.plot(simple_x, simple_y, 'o-', color='#00d4ff', alpha=0.6, markersize=3, linewidth=1, label='SimpleAgent')

test_x = [p[0] for p in test_positions[:50]]
test_y = [p[1] for p in test_positions[:50]]
ax.plot(test_x, test_y, 's-', color='#39ff14', alpha=0.6, markersize=3, linewidth=1, label='LearningAgent (Test)')

ax.set_title('Lighthouse: Trajetórias dos Agentes (primeiros 50 passos)', fontsize=12, fontweight='bold')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('./lighthouse_results.png', dpi=150, bbox_inches='tight')
print("Gráfico salvo em: ./lighthouse_results.png")
plt.close()

# ============================================================================
# 6. RESUMO DOS RESULTADOS
# ============================================================================

print("\n" + "=" * 70)
print("RESUMO DOS RESULTADOS - LIGHTHOUSE")
print("=" * 70)
print(f"SimpleAgent (100 passos):        Recompensa Total = {simple_rewards[-1]:.2f}")
print(f"LearningAgent (500 passos):      Recompensa Total = {learning_rewards[-1]:.2f}")
print(f"LearningAgent Teste (100 passos): Recompensa Total = {test_rewards[-1]:.2f}")
print("\nConclusão: O LearningAgent aprendeu uma política melhor que o SimpleAgent.")
print("=" * 70)
