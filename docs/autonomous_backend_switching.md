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
