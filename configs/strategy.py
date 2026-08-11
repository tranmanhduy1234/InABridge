import json
from pathlib import Path

config = {
    "schema_version": "1.0",
    "name": "instructblip_three_stage_training_plan",
    "description": (
        "Three-stage training curriculum for an InstructBLIP-style VLM. "
        "Stage 1 learns visual-language representations in the Q-Former; "
        "Stage 2 aligns the Q-Former output with the frozen LLM through the projector; "
        "Stage 3 performs instruction adaptation with QLoRA while keeping the Q-Former frozen."
    ),
    "global_policy": {
        "image_encoder": {
            "trainable": False,
            "reason": "Keep the pretrained visual representation stable."
        },
        "qformer_protection": {
            "principle": (
                "Q-Former is the learned visual-language representation interface. "
                "After Stage 1 it should only be adjusted very conservatively in Stage 2b, "
                "then frozen for all of Stage 3."
            ),
            "stage3_trainable": False
        },
        "lr_relationship_stage3": "qlora_lr > projector_lr >> qformer_lr=0"
    },
    "stages": [
        {
            "stage": 1,
            "name": "representation_alignment",
            "goal": "Learn visual-language representation inside the Q-Former.",
            "default_max_steps": 250000,
            "objectives": {
                "ITC": {
                    "enabled": True,
                    "name": "image_text_contrastive"
                },
                "ITM": {
                    "enabled": True,
                    "name": "image_text_matching"
                },
                "ITG": {
                    "enabled": True,
                    "name": "image_grounded_text_generation"
                }
            },
            "modules": {
                "image_encoder": {
                    "trainable": False,
                    "lr": 0.0
                },
                "qformer": {
                    "trainable": True,
                    "lr": {
                        "start": 0.0001,
                        "end": 0.00003,
                        "schedule": "cosine"
                    }
                },
                "projector": {
                    "enabled": False,
                    "trainable": False,
                    "lr": 0.0
                },
                "llm_base": {
                    "enabled": False,
                    "trainable": False,
                    "lr": 0.0
                },
                "qlora": {
                    "enabled": False,
                    "trainable": False,
                    "lr": 0.0
                }
            }
        },
        {
            "stage": 2,
            "name": "modality_bridge_alignment",
            "goal": (
                "Stabilize and align the bridge from Q-Former latent space "
                "to the frozen LLM embedding space."
            ),
            "objective": {
                "type": "causal_language_modeling",
                "name": "bridge_lm_loss"
            },
            "subphases": [
                {
                    "name": "stage_2a_projector_warmup",
                    "default_max_steps": 1000,
                    "goal": (
                        "Warm up the randomly initialized projector without allowing "
                        "the Q-Former or LLM to move."
                    ),
                    "modules": {
                        "image_encoder": {
                            "trainable": False,
                            "lr": 0.0
                        },
                        "qformer": {
                            "trainable": False,
                            "lr": 0.0
                        },
                        "projector": {
                            "trainable": True,
                            "lr": 0.0001
                        },
                        "llm_base": {
                            "trainable": False,
                            "lr": 0.0
                        },
                        "qlora": {
                            "enabled": False,
                            "trainable": False,
                            "lr": 0.0
                        }
                    }
                },
                {
                    "name": "stage_2b_joint_bridge_alignment",
                    "default_max_steps": 10000,
                    "goal": (
                        "Allow only a very small Q-Former adjustment while the projector "
                        "learns the main mapping into the LLM-compatible representation."
                    ),
                    "modules": {
                        "image_encoder": {
                            "trainable": False,
                            "lr": 0.0
                        },
                        "qformer": {
                            "trainable": True,
                            "lr": {
                                "start": 0.000005,
                                "end": 0.000001,
                                "schedule": "cosine"
                            }
                        },
                        "projector": {
                            "trainable": True,
                            "lr": {
                                "start": 0.0001,
                                "end": 0.00005,
                                "schedule": "cosine"
                            }
                        },
                        "llm_base": {
                            "trainable": False,
                            "lr": 0.0
                        },
                        "qlora": {
                            "enabled": False,
                            "trainable": False,
                            "lr": 0.0
                        }
                    }
                }
            ]
        },
        {
            "stage": 3,
            "name": "instruction_adaptation",
            "goal": (
                "Adapt multimodal behavior through QLoRA while preserving the "
                "Q-Former visual-language representation learned earlier."
            ),
            "objective": {
                "type": "causal_language_modeling",
                "name": "instruction_ce_loss"
            },
            "qformer_policy": {
                "trainable": False,
                "lr": 0.0,
                "reason": (
                    "Stage 1 already learned visual-language representation and Stage 2 "
                    "already allowed the only intended small alignment adjustment."
                )
            },
            "subphases": [
                {
                    "name": "stage_3a_qlora_warmup",
                    "default_max_steps": 1000,
                    "goal": (
                        "Introduce QLoRA conservatively while keeping the Q-Former fully frozen."
                    ),
                    "modules": {
                        "image_encoder": {
                            "trainable": False,
                            "lr": 0.0
                        },
                        "qformer": {
                            "trainable": False,
                            "lr": 0.0
                        },
                        "projector": {
                            "trainable": True,
                            "lr": {
                                "start": 0.00005,
                                "end": 0.00001,
                                "schedule": "cosine"
                            }
                        },
                        "llm_base": {
                            "trainable": False,
                            "quantization": "4bit"
                        },
                        "qlora": {
                            "enabled": True,
                            "trainable": True,
                            "lr": 0.00001
                        }
                    }
                },
                {
                    "name": "stage_3b_full_instruction_tuning",
                    "default_max_steps": 79000,
                    "goal": (
                        "Shift most downstream adaptation responsibility to QLoRA. "
                        "Q-Former remains frozen permanently."
                    ),
                    "modules": {
                        "image_encoder": {
                            "trainable": False,
                            "lr": 0.0
                        },
                        "qformer": {
                            "trainable": False,
                            "lr": 0.0
                        },
                        "projector": {
                            "trainable": True,
                            "lr": 0.00001
                        },
                        "llm_base": {
                            "trainable": False,
                            "quantization": "4bit"
                        },
                        "qlora": {
                            "enabled": True,
                            "trainable": True,
                            "lr": {
                                "start": 0.00005,
                                "end": 0.00002,
                                "schedule": "cosine"
                            }
                        }
                    }
                }
            ]
        }
    ],
    "implementation_notes": {
        "default_steps_are_tunable": True,
        "stage2a": "1000 steps is a conservative default for projector warm-up.",
        "stage2b": "Keep Q-Former LR at least ~20x lower than the projector LR.",
        "stage3": (
            "Do not unfreeze Q-Former again. Downstream instruction adaptation should "
            "primarily be absorbed by QLoRA, with the projector using a smaller LR."
        )
    }
}

out_path = Path("/home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/configs/instructblip_three_stage_training_plan.json")
with out_path.open("w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print(f"Created: {out_path}")