from gymnasium.envs.registration import register

register(
    id='CustomTaxi-v0',
    entry_point='environments.CrashingTaxiEnv:CustomTaxiEnv',
)