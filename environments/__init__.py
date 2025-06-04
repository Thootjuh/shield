from gymnasium.envs.registration import register

register(
    id='CustomTaxi-v0',
    entry_point='environments.CrashingTaxiEnv:CustomTaxiEnv',
)
register(
    id='FrozenLakeCustom-v0',
    entry_point='environments.frozenLakeEnv:FrozenLakeEnv',
)