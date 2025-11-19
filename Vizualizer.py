import tkinter as tk
import numpy as np

class Vizualizer:
    def __init__(self, environment, agents, cell_size=40):
        self.env = environment
        self.agents = agents
        self.cell_size = cell_size

        h, w = self.env.grid.shape

        self.window = tk.Tk()
        self.window.title("Simulador SMA")

        self.canvas = tk.Canvas(self.window, width=w * cell_size, height=h * cell_size, bg="white")
        self.canvas.pack()

    def draw_grid(self):
        h, w = self.env.grid.shape
        cs = self.cell_size

        for y in range(h):
            for x in range(w):
                val = self.env.grid[y, x]

                color = "white"
                if val == 9:
                    color = "yellow"
                elif val > 0:
                    color = "#66ff66"

                self.canvas.create_rectangle(
                    x*cs, y*cs, x*cs+cs, y*cs+cs,
                    fill=color, outline="gray"
                )

    def draw_agents(self):
        cs = self.cell_size
        for agent, pos in self.env.agent_positions.items():
            x, y = pos
            self.canvas.create_oval(
                x*cs+8, y*cs+8, x*cs+cs-8, y*cs+cs-8,
                fill="blue"
            )
            self.canvas.create_text(x*cs+cs/2, y*cs+cs/2, text=agent.name[0], fill="white")

    def update(self):
        self.canvas.delete("all")
        self.draw_grid()
        self.draw_agents()
        self.canvas.update()

    def start(self):
        """Mantém a janela aberta no macOS."""
        self.window.mainloop()
