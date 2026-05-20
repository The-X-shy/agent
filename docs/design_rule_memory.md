# Design Rule Memory

DesignRule memory is a projection from claims, artifacts, and traces. It is not a source of truth.

## Compile

`DesignRuleManager.compile_from_claims()` compares supported claims and artifact metrics across baseline runs. In the current mock setting, it can compile:

```text
controlled chromatic EDOF gives better joint depth-spectral tradeoff than fully achromatic mock encoder under current mock setting.
```

## Contradictions

`DesignRuleManager.detect_contradictions()` checks claims that conflict with metric evidence. For example:

```text
achromatic encoder is best for spectral separability
```

If controlled chromatic EDOF has higher `spectral_separability`, the old claim is marked `contradicted` or lowered to partial support.

## Explain

```bash
python -m optiresearch.cli explain-rule --rule-id <rule_id>
```

The explanation includes rule status, confidence, supporting claims, supporting artifacts, metrics, and source traces.
