# Модель: Оптимальне керування процесом очищення водойми
# Автор: Маляренко Анастасія в групі з Брагар Софією, група АІ-233
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

class WaterOptimizer:
    def __init__(self, V, Q, k, C_initial, C_target, Cin_max):
        """Ініціалізація параметрів системи"""
        self.V = V; self.Q = Q; self.k = k
        self.C_initial = C_initial
        self.C_target = C_target
        self.Cin_max = Cin_max

    def cstr_ode(self, t, C, Cin_profile, t_points):
        """Права частина ЗДР (динаміка концентрації)"""
        u_t = np.interp(t, t_points, Cin_profile)
        dCdt = (self.Q / self.V) * u_t - (self.Q / self.V + self.k) * C
        return dCdt

    def objective_minimum_effort(self, Cin_profile, T_fix, N_steps):
        """Функція цілі J_E: Мінімізація зусилля/витрат"""
        t_points = np.linspace(0, T_fix, N_steps)
        sol = solve_ivp(
            self.cstr_ode,
            (0, T_fix),
            [self.C_initial],
            args=(Cin_profile, t_points),
            method='RK45',
            t_eval=t_points
        )
        C_final = sol.y[0, -1]
        penalty = 0 if C_final <= self.C_target else (C_final - self.C_target)**2 * 1e5
        dt = T_fix / (N_steps - 1)
        effort_cost = np.sum(Cin_profile**2) * dt
        return effort_cost + penalty

    def find_optimal_control(self, T_fix, N_steps=50):
        """Запуск оптимізації SLSQP"""
        initial_guess = np.ones(N_steps) * self.Cin_max
        bounds = [(0, self.Cin_max)] * N_steps
        result = minimize(
            self.objective_minimum_effort,
            initial_guess,
            args=(T_fix, N_steps),
            method='SLSQP',
            bounds=bounds
        )
        return result.x, result.fun

if __name__ == "__main__":
    # Тестовий запуск моделі при старті контейнера
    print("=== Запуск моделі очищення водойми ===")
    optimizer = WaterOptimizer(V=1000, Q=50, k=0.01, C_initial=0.5, C_target=0.1, Cin_max=0.2)
    control, cost = optimizer.find_optimal_control(T_fix=10)
    print("Оптимізацію завершено успішно!")
    print(f"Мінімальні витрати енергії/зусиль: {cost:.4f}")
