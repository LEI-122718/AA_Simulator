from Agent import *
from Environment import *

class MotorSimulacao:
    """
    Motor simples de simulação com atributos fixos.
    Controla o ciclo: observação -> ação -> atualização.
    """

    def __init__(self):
        self.tempo = 0

        # Atributos fixos iniciais
        self.ambiente = None
        self.agentes = []

    # ------------------------------
    # cria()
    # ------------------------------
    def cria(self, ficheiro_param: str):
        """
        Nesta versão simples, ignora o ficheiro e cria tudo fixo.
        """
        print("Criando ambiente e agentes (fixos).")

        # Criar ambiente fixo (exemplo)
        self.ambiente = Environment(width=10, height=10, self.agentes)

        # Criar agentes fixos
        agente1 = Agent( 2, 3,"A1")
        agente2 = Agent( 7, 5,"A1")

        self.agentes.append(agente1)
        self.agentes.append(agente2)

        # Registar no ambiente
        self.ambiente.add_agent(agente1)
        self.ambiente.add_agent(agente2)

        print("Configuração fixa criada.")

    # ------------------------------
    # listaAgentes()
    # ------------------------------
    def listaAgentes(self):
        """Retorna a lista de agentes."""
        return self.agentes

    # ------------------------------
    # executa()
    # ------------------------------
    def executa(self):
        """Executa a simulação durante 5 passos (fixo)."""
        print("Executando simulação por 5 passos...")

        for _ in range(5):
            self.passoSimulacao()

        print("Simulação terminada.")

    # ------------------------------
    # passoSimulacao()
    # ------------------------------
    def passoSimulacao(self):
        """Executa um ciclo de simulação."""
        self.tempo += 1
        print(f"\n--- PASSO {self.tempo} ---")

        # 1. Observações
        observacoes = {
            ag: self.ambiente.observacao_para(ag)
            for ag in self.agentes
        }

        # 2. Ações
        acoes = {
            ag: ag.age(observacoes[ag])
            for ag in self.agentes
        }

        # 3. Executar ações
        for agente, acao in acoes.items():
            self.ambiente.agir(acao, agente)

        # 4. Atualizar ambiente
        self.ambiente.atualizacao()

    def add_agent(self, agent):
        self.agents.append(agent)
