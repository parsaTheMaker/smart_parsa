# DeAL Research Workflow

Use the DeAL Codex skills below when their trigger matches the task:

| Task | Skill |
| --- | --- |
| LaTeX, ICLR writing, citations, PDF, or ShareLaTeX | `deal-paper-production` |
| Results, checkpoints, metrics, configurations, or comparison claims | `deal-experiment-audit` |
| VTP/VTK, scientific figures, field panels, or interactive views | `deal-scientific-visualization` |
| CUDA, DDP, worker allocation, inference throughput, or job recovery | `deal-gpu-operations` |
| PDE data generation, meshing, remeshing, or physics documentation | `deal-physics-benchmark` |

Shared invariants:

- Preserve raw artifacts and trace reported values to an evaluator and checkpoint.
- Do not change a metric definition, physical condition, or figure normalization without checking the matching code and rendered output.
- Prefer a small smoke test before long GPU, meshing, or rendering jobs.
- Keep generated build artifacts and interpreter caches out of version-control commits.
- Use the existing project visual language unless the task explicitly calls for a redesign.
