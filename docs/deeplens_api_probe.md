# DeepLens API Probe

`probe-deeplens-api` inspects the installed `deeplens` package without assuming a fixed API.

It records:

- import availability;
- DeepLens version and import path;
- discovered modules, classes, and functions;
- candidate lens, surface, phase/DOE, and optimization APIs;
- structured errors for missing modules.

Command:

```bash
python -m optiresearch.cli probe-deeplens-api
```

Outputs:

- `workspace/reports/deeplens_api_probe.json`
- `workspace/reports/deeplens_api_probe.md`

Missing DeepLens is reported as `available=false`; it is not a crash.
