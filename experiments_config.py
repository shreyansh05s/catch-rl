
experiments = {
    "Baseline": [
        {
            "group": "Baseline",
            "job_type": "A2C",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default"
        },
        {
            "group": "Baseline",
            "job_type": "A2C-Bootstrap",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "bootstrap": True,
                "n_steps": 100
            }
        },
        {
            "group": "Baseline",
            "job_type": "A2C-Baseline",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "baseline": True
            }
        },
        {
            "group": "Baseline",
            "job_type": "A2C-Bootstrap-Baseline",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "baseline": True,
                "bootstrap": True,
                "n_steps": 100
            }
        },
        {
            "group": "Baseline",
            "job_type": "Reinforce-baseline",
            "model": "Reinforce",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "lr": 0.001,
                "gamma": 0.85,
                "hidden_size": 128,
                "num_episodes": 1000,
                "beta": 0.01,
                "entropy_reg": True,
                "max_steps": 250
            }
        },
        {
            "group": "Baseline",
            "job_type": "Reinforce-baseline2",
            "model": "Reinforce",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "lr": 0.001,
                "gamma": 0.7,
                "hidden_size": 128,
                "num_episodes": 1000,
                "beta": 0.01,
                "entropy_reg": True,
                "max_steps": 250,
                "lr_scheduler": False
            }
        }

    ],
    "Entropy": [
        {
            "group": "Entropy",
            "job_type": "A2C-entropy-0.01",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "entropy_weight": 0.01,
                "entropy_regularization": True
            }
        },
        {
            "group": "Entropy",
            "job_type": "A2C-entropy-0.1",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "entropy_weight": 0.1,
                "entropy_regularization": True
            }
        },
        {
            "group": "Entropy",
            "job_type": "A2C-entropy-0.3",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "entropy_weight": 0.3,
                "entropy_regularization": True
            }
        },
        {
            "group": "Entropy",
            "job_type": "A2C-entropy-0.7",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "entropy_weight": 0.7,
                "entropy_regularization": True
            }
        },
        {
            "group": "Entropy",
            "job_type": "A2C",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "entropy_regularization": False
            }
        },
        {
            "group": "Entropy",
            "job_type": "Reinforce-entropy-0.001",
            "model": "Reinforce",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "beta": 0.001,
                "entropy_reg": True
            }
        }
    ],
    "LearningRate": [
        {
            "group": "LearningRate",
            "job_type": "A2C-lr-1e-2",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "lr": 1e-2,
                "lr_scheduler": False
            }
        },
        {
            "group": "LearningRate",
            "job_type": "A2C-lr-1e-3",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "lr": 1e-3,
                "lr_scheduler": False
            }
        },
        {
            "group": "LearningRate",
            "job_type": "A2C-lr-1e-4",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "lr": 1e-4,
                "lr_scheduler": False
            }
        },
        {
            "group": "LearningRate",
            "job_type": "A2C-lr-1e-2-lr_scheduler",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "lr": 1e-2,
                "lr_scheduler": True
            }
        },
        {
            "group": "LearningRate",
            "job_type": "A2C-lr-1e-3-lr_scheduler",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "lr": 1e-3,
                "lr_scheduler": True
            }
        }
    ],
    "LearningRate_Reinforce": [
        {
            "group": "LearningRate_Reinforce",
            "job_type": "Reinforce-lr-1e-2-scheduler",
            "model": "Reinforce",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "gamma": 0.85,
                "hidden_size": 128,
                "num_episodes": 1000,
                "beta": 0.01,
                "entropy_reg": True,
                "max_steps": 250,
                "lr": 0.01,
                "lr_scheduler": True
            }
        },
        {
            "group": "LearningRate_Reinforce",
            "job_type": "Reinforce-lr-1e-3-scheduler",
            "model": "Reinforce",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "gamma": 0.85,
                "hidden_size": 128,
                "num_episodes": 1000,
                "beta": 0.01,
                "entropy_reg": True,
                "max_steps": 250,
                "lr": 0.001,
                "lr_scheduler": True
            }
        },
        {
            "group": "LearningRate_Reinforce",
            "job_type": "Reinforce-lr-1e-3",
            "model": "Reinforce",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "lr": 0.001,
                "gamma": 0.85,
                "hidden_size": 128,
                "num_episodes": 1000,
                "beta": 0.01,
                "entropy_reg": True,
                "max_steps": 250,
                "lr_scheduler": False
            }
        },
        {
            "group": "LearningRate_Reinforce",
            "job_type": "Reinforce-lr-1e-4",
            "model": "Reinforce",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "lr": 0.0001,
                "gamma": 0.85,
                "hidden_size": 128,
                "num_episodes": 1000,
                "beta": 0.01,
                "entropy_reg": True,
                "max_steps": 250,
                "lr_scheduler": False
            }
        }
    ],
    "Entropy-Reinforce": [

        {
            "group": "Entropy-Reinforce",
            "job_type": "Reinforce-entropy-0.001",
            "model": "Reinforce",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "lr": 0.001,
                "gamma": 0.85,
                "hidden_size": 128,
                "num_episodes": 1000,
                "beta": 0.001,
                "entropy_reg": True,
                "max_steps": 250
            }
        },
        {
            "group": "Entropy-Reinforce",
            "job_type": "Reinforce-entropy-0.01",
            "model": "Reinforce",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "lr": 0.001,
                "gamma": 0.85,
                "hidden_size": 128,
                "num_episodes": 1000,
                "beta": 0.01,
                "entropy_reg": True,
                "max_steps": 250
            }
        },
        {
            "group": "Entropy-Reinforce",
            "job_type": "Reinforce-entropy-0.1",
            "model": "Reinforce",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "lr": 0.001,
                "gamma": 0.85,
                "hidden_size": 128,
                "num_episodes": 1000,
                "beta": 0.1,
                "entropy_reg": True,
                "max_steps": 250
            }
        },
        {
            "group": "Entropy-Reinforce",
            "job_type": "Reinforce-entropy-0.3",
            "model": "Reinforce",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "lr": 0.001,
                "gamma": 0.85,
                "hidden_size": 128,
                "num_episodes": 1000,
                "beta": 0.3,
                "entropy_reg": True,
                "max_steps": 250
            }
        },
        {
            "group": "Entropy-Reinforce",
            "job_type": "Reinforce-entropy-0.7",
            "model": "Reinforce",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "lr": 0.001,
                "gamma": 0.85,
                "hidden_size": 128,
                "num_episodes": 1000,
                "beta": 0.1,
                "entropy_reg": True,
                "max_steps": 250
            }
        },
        {
            "group": "Entropy-Reinforce",
            "job_type": "Reinforce-no-entropy",
            "model": "Reinforce",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "lr": 0.001,
                "gamma": 0.85,
                "hidden_size": 128,
                "num_episodes": 1000,
                "beta": 0.1,
                "entropy_reg": False,
                "max_steps": 250
            }
        }

    ],
    "ObservationType": [
        {
            "group": "ObservationType",
            "job_type": "A2C-pixel",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default"
        },
        {
            "group": "ObservationType",
            "job_type": "A2C-vector",
            "model": "ActorCritic",
            "observation_type": "vector",
            "env": "default"
        }
    ],
    "A2C-Bootstrap": [
        {
            "group": "A2C-Bootstrap",
            "job_type": "A2C-Bootstrap-10",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "bootstrap": True,
                "n_steps": 10
            }
        },
        {
            "group": "A2c-Bootstrap",
            "job_type": "A2C-Bootstrap-40",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "bootstrap": True,
                "n_steps": 40
            }
        },
        {
            "group": "A2c-Bootstrap",
            "job_type": "A2C-Bootstrap-100",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "bootstrap": True,
                "n_steps": 100
            }
        },
        {
            "group": "A2c-Bootstrap",
            "job_type": "A2C-Bootstrap-200",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "hyperparameter": {
                "bootstrap": True,
                "n_steps": 200
            }
        }
    ],
    "Env-size": [
        {
            "group": "Env-size",
            "job_type": "A2C-Env-7X7",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "rows": 7,
            "columns": 7,
            "normalize_graph": True
        },
        {
            "group": "Env-size",
            "job_type": "A2C-Env-10X10",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "rows": 10,
            "columns": 10,
            "normalize_graph": True
        },
        {
            "group": "Env-size",
            "job_type": "A2C-Env-25X25",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default3",
            "rows": 25,
            "columns": 25,
            "normalize_graph": True
        },
        {
            "group": "Env-size",
            "job_type": "A2C-Env-14X7",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default4",
            "rows": 14,
            "columns": 7,
            "speed": 1.0,
            "normalize_graph": True
        },
        {
            "group": "Env-size",
            "job_type": "A2C-Env-7X14",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default4",
            "rows": 7,
            "columns": 14,
            "speed": 1.0,
            "normalize_graph": True
        }
    ],
    "Env-speed": [
        {
            "group": "Env-speed",
            "job_type": "A2C-1.0",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "speed": 1.0,
            "normalize_graph": True
        },
        {
            "group": "Env-speed",
            "job_type": "A2C-0.5",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "speed": 0.5,
            "normalize_graph": True
        },
        {
            "group": "Env-speed",
            "job_type": "A2C-2.0",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "speed": 2.0,
            "normalize_graph": True
        },
        {
            "group": "Env-speed",
            "job_type": "A2C-5.0",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "speed": 5.0,
            "normalize_graph": True
        }
    ],
    "Env-variations": [
        {
            "group": "Env-variations",
            "job_type": "Env-14X7-speed-0.5",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default",
            "rows": 14,
            "columns": 7,
            "speed": 0.5,
            "normalize_graph": True
        },
        {
            "group": "Env-variations",
            "job_type": "Env-7X7-speed-0.5-vector",
            "model": "ActorCritic",
            "observation_type": "vector",
            "env": "default2",
            "rows": 7,
            "columns": 7,
            "speed": 0.5,
            "normalize_graph": True
        },
        {
            "group": "Env-variations",
            "job_type": "max-misses-20-max-steps-300",
            "model": "ActorCritic",
            "observation_type": "pixel",
            "env": "default3",
            "rows": 7,
            "columns": 7,
            "max_misses": 20,
            "max_steps": 300,
            "normalize_graph": True
        },
        {
            "group": "Env-variations",
            "job_type": "Env-14X7-speed-0.5-vector",
            "model": "ActorCritic",
            "observation_type": "vector",
            "env": "default2",
            "rows": 14,
            "columns": 7,
            "speed": 0.5,
            "normalize_graph": True
        },
        {
            "group": "Env-variations",
            "job_type": "Env-7X7-speed-2-vector",
            "model": "ActorCritic",
            "observation_type": "vector",
            "env": "default2",
            "rows": 7,
            "columns": 7,
            "speed": 2,
            "normalize_graph": True
        }
    ]
}
