import numpy as np

class OUProcess:
    """
    Ornstein-Uhlenbeck sensor drift:
    dX = θ(μ - X)dt + σ dW
    Discretized: X_{t+1} = X_t + θ(μ - X_t)dt + σ * sqrt(dt) * N(0,1)
    """
    def __init__(self, mu: float, theta=0.1, sigma=0.02, dt=1.0):
        self.mu = mu          # true value / mean
        self.theta = theta    # reversion speed
        self.sigma = sigma    # noise volatility
        self.dt = dt          # time step
        self.x = mu           # start at true value

    def step(self) -> float:
        noise = np.random.normal(0, 1)
        self.x += (self.theta * (self.mu - self.x) * self.dt
                   + self.sigma * np.sqrt(self.dt) * noise)
        return round(self.x, 4)

    def update_mean(self, new_mu: float):
        """Call this when the true value changes (e.g. pH after dosing)."""
        self.mu = new_mu