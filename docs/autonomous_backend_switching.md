# Autonomous Backend Switching

## Overview

When the autonomous loop reaches `claim_ceiling_reached` on the current backend,
it can automatically switch to a higher-evidence backend instead of stopping.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `allow_backend_switching` | `True` | Enable automatic backend switching |
| `max_backend_switches` | `1` | Maximum number of backend switches per loop |

## How It Works

1. The trajectory evaluator detects `claim_ceiling_reached` (same backend, same ceiling, 2+ iterations).

2. The loop queries `get_next_backend()` from the backend progression graph.

3. If a valid next backend is found and it's in `allowed_backends`, the loop:
   - Updates the active `backend_id`
   - Increments `backend_switch_count`
   - Marks the current iteration as `switch_backend` action
   - Continues to the next iteration with the new backend

4. The loop stops only when:
   - `max_backend_switches` is reached
   - No next backend exists in the progression graph
   - Next backend is not in `allowed_backends`
   - `max_iterations` is reached

## Post-Switch Validation (Phase 31)

After a backend switch, the loop runs a lightweight backend probe on the new
backend before proceeding with full experiments:

5. At switch time, `pending_backend_switch=True` is injected into the
   iteration's `execution_result` along with `switched_from_backend` and
   `switched_to_backend`.

6. The next iteration's StrategyEngine sees `pending_backend_switch` and
   recommends `probe_new_backend`, which maps to the `backend_probe` task type.

7. `run_lightweight_backend_probe()` executes a fast (<5s) FFT PSF probe to
   validate backend availability. For DeepLens backends, it first checks
   whether `deeplens.geolens` is importable.

8. If the probe succeeds: `backend_switch_validated=True`,
   `pending_backend_switch` is cleared, and the loop continues with normal
   experiments on the new backend.

9. If the probe fails and alternatives exist: the loop tries the next
   available edge from the progression graph via `get_all_edges_from()`.

## Alternative Backend Fallback

When the primary next backend's probe fails:

```python
from optiresearch.backends.progression import get_all_edges_from
alt_edges = get_all_edges_from(source_backend)
# Try each edge, skipping the failed one
# Respects max_backend_switches limit
# Falls back to stop_reason=backend_switch_failed if all exhausted
```

## CLI Usage

```bash
python -m optiresearch.cli run-autonomous-research-loop-v2 \
  --objective "continue after claim ceiling by switching backend" \
  --max-iterations 3 \
  --execution-mode local \
  --allowed-backends "phase_to_fft_proxy,deeplens_geolens_geometric" \
  --allow-backend-switching \
  --max-backend-switches 1
```
