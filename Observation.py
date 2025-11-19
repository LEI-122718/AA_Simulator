class Observation:
    """
    Observação devolvida pelo ambiente.
    Pode incluir direção do farol, valores das células, vizinhança, etc.
    """
    def __init__(self, info: dict):
        self.info = info

    def get(self, key, default=None):
        return self.info.get(key, default)

    def __repr__(self):
        return f"Observation({self.info})"
