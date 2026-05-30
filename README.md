# OptiResearch Agent

OptiResearch Agent 是一个面向计算光学研究的 MVP 后端骨架。当前版本使用规则化 agent、MockDeepLensAdapter、SQLite 和本地 artifact store，跑通从研究目标到仿真、记忆、证据审查的最小闭环。

Phase 2 增加了标准实验规范、可解释 claim-evidence、artifact inspection、PlanTemplate / SkillMemory 编译，以及 OptiMemoryBench 初版。

Phase 3 增加了 5 类 mock optical encoder baseline、DesignRule memory、contradiction 检测，以及 OptiMemoryBench 消融模式。

Phase 4 冻结了 `ExperimentSpec v0.1`，新增真实 DeepLens adapter 契约、统一 adapter 返回格式、论文实验协议，以及 paper summary / evidence table 导出。

Phase 5 增加真实 DeepLens 的最小接入闭环：环境探测、`ExperimentSpec v0.1` 到 `DeepLensCandidateConfig` 的转换、真实 smoke run 入口，以及 `run-mvp --backend` 后端选择。

Phase 6 增加真实 DeepLens baseline 协议、mock-real alignment report、DeepLens capability model、smoke-level claim 降级，以及 Phase 6 论文材料导出。

Phase 7 增加 DeepLens encoder proxy strategy 注册表、5 类 encoder 的 proxy transform、encoder manifest 与 realization level tracking。

Phase 8 增加半原生 (semi-native) DeepLens 接入、LLM 集成（DeepSeek 接口），以及 Phase 8 报告。

Phase 9 增加合成 HSI reconstruction pipeline：数据集、前向模型、线性重建 baseline、HSI metrics、ClaimEvidence。

Phase 10 增加 optical-sensitive HSI reconstruction benchmark：混合材料数据集、optical-sensitive 前向模型、OpticalFeatureExtractor、OpticalConditionedLinearReconstructor、TinyCNN 可选后端、encoder 重建 ranking、Phase 10 报告。

Phase 11 增加 public/local HSI dataset adapter、TinyCNN/UNet optional reconstructor contract、optical feature maps、HSI matrix、matrix-level ClaimEvidence、DesignRule 编译，以及 Phase 11 报告。Public datasets 只支持本地路径，不自动下载；mock optical encoder 的 synthetic/public 结果不能写成真实相机实验。

Phase 12 增强 local/public HSI 数据接入、CAVE/ICVL 本地扫描、DeepLens wavelength-aware PSF contract、public HSI matrix、public dataset ClaimEvidence scope，并冻结论文实验协议 v0.1。

Phase 13 冻结 paper-ready benchmark、生成 10 张论文表格、建立 claim whitelist/blacklist、统计 evidence distribution、生成 warnings audit、导出最终论文证据包和 Phase 13 报告。

Phase 17 增加 remote WSL worker execution：Mac 端负责 controller、memory、evidence、report，WSL 端负责真实 DeepLens source smoke 与严格 DeepLens-backed co-design。远程命令必须经过 allowlist，失败返回结构化错误，fallback 不会写成 DeepLens-backed claim。

Phase 18–23 增加原生可微优化路径：DeepLens surface 原生优化 probe (Phase 19)、可微 HSI proxy co-design (Phase 20)、完整可微 HSI reconstruction co-design (Phase 21)、wave-optics native path probe (Phase 22)、stable native lens HSI co-design with rollback (Phase 23)。

Phase 24 升级为 Agentic Differentiable Optics Framework：统一 backend registry (8 backends)、ExperimentControllerV2 (统一实验入口)、StrategyEngine (自动策略推荐)、ResearchMemoryV2 (Phase 18-23 经验沉淀)、ClaimGateV2 (声明预检 8 种违规类型)、Optical Objective Library (可组合 loss)、Autograd Auditor (自动求导链路检测)、Agent System Report (系统报告生成)。

Phase 25 增加闭环自主研究 Agent：将 Phase 24 组件组合为 autonomous research loop (strategy → plan → execute → diagnose → claim gate → memory → decide)。支持 dry_run / local / remote_opt_in 三种执行模式，默认 dry_run，remote 需显式 opt-in。新增 Strategy-to-Spec compiler、Trajectory Evaluator、Autonomous Loop Report、远程安全 guard、Claim Gate 硬强制执行。

Phase 26 增加 LLM-assisted autonomous research planner：引入 LLMPlanner 基于 ResearchMemoryV2、BackendRegistry 和 recent results 生成候选研究计划。支持 mock / deepseek provider。LLM proposal 经过 PlannerValidator (10 项安全检查)、ClaimGateV2 (8 种违规检测)、schema validation。保留 rule-based StrategyEngine 作为 fallback。支持 plan-with-llm CLI、planner trace 审计、LLM planner report。

Phase 27 增加真实 LLM 自主循环验证：使用真实 DeepSeek provider 端到端验证 LLM-assisted autonomous research loop。新增 provider 环境检查、真实 DeepSeek planner smoke test、planner 鲁棒性测试（10 种非法 LLM 输出场景）、trace 脱敏（API key / Authorization header / env value 三重脱敏）、LLM provider validation report。所有真实 LLM 测试需显式 opt-in（OPTIRESEARCH_ENABLE_REAL_LLM_TESTS=1）。

Phase 28 增加 LLM 规划多轮本地实验执行：新增 executable LLM planning mode（引导 LLM 优先选择可执行策略）、pure-PyTorch FFT 轻量实验（无需 DeepLens）、多轮 feedback context（将上一轮标量指标压缩为 LLM 可读上下文）、enhanced trajectory report（6 个新 section）。支持 prefer_executable_actions 和 lightweight_psf_probe task type。真实 DeepSeek 多轮本地 loop 通过测试。

Phase 29 修复 claim-downgrade 早停并启用多轮 metric trajectory：trajectory evaluator 新增 min_iterations_before_stop 和 no_improvement_patience 防止 premature stop；experiment controller 使用 backend task evidence cap 替代 claim ceiling 阻断；新增 lightweight stable lens HSI 路由；LLM prompt 新增 metric trajectory 约束；metrics schema 统一化；spec_patch 安全过滤；report 增强（claim downgrade events、metric trajectory data、stop diagnostics、spec patch table）。支持 multi-iteration autonomous loop with metric-first evaluation。

Phase 30 增加多后端自主切换：新增 BackendProgressionGraph（7 条默认 progression edge）；StrategyEngine 新增 claim_ceiling_reached 自动推荐 switch_backend_after_claim_ceiling；trajectory evaluator 新增 backend_history / backend_switch_count / evidence_level_progression 跨后端追踪；autonomous loop 在 claim_ceiling_reached 后自动查询 progression graph 切换 backend 继续执行；LLM prompt 新增 backend switching 指引；controller 新增 psf_probe 和 component_optimization 轻量路由；report 新增 Backend Progression 和 Evidence Level Progression section。

Phase 31 新增后端切换后验证：post-switch strategy context 注入（pending_backend_switch / switched_from_backend / switched_to_backend）；StrategyEngine 新增 probe_new_backend 最高优先级规则；新增 backend_probe task type 和 lightweight backend probe runtime（支持 DeepLens unavailable 结构化返回）；strategy-to-spec 映射 probe_new_backend → backend_probe；LLM prompt 新增 Rule 17 和 pending switch context 动态 section；backend progression graph 新增 get_all_edges_from() 支持 alternative fallback；trajectory evaluator 新增 backend_switch_triggered / validated / probe_success / unavailable / evidence_gain_after_switch；report 新增 Backend Probe Results 和 Backend Switch Validation section；autonomous loop 在 probe 失败时自动尝试 alternative backend 或返回 structured unavailable。

Phase 32 新增后端切换后延续实验：post-probe continuation signal（post_probe_continuation_required / validated_backend_id / validated_backend_evidence_level）；StrategyEngine 新增 run_validated_backend_experiment 规则；新增 native_lens_simulation_codesign task type；deep GeoLens geometric PSF probe（实际调用 deeplens.geolens.psf_geometric 路径，验证 differentiability 和 gradient flow）；CLI 注册 run-lightweight-backend-probe 命令（支持 shallow/deep probe depth）；alternative fallback 增强为 try-all-edges（遍历所有 alternative edges 直到成功或耗尽）；LLM prompt 新增 Rule 18 和 post-probe continuation 动态 section；report 新增 Post-Probe Continuation 和 Alternative Backend Attempts section。

Phase 33 新增完整 DeepLens native GeoLens HSI co-design path：native_lens_simulation_codesign 在 deeplens_geolens_geometric 后端下路由到 _run_stable_lens_hsi（完整三阶段训练：warmup / joint finetune / final adaptation）；新增 execution_fidelity 字段区分 lightweight_proxy 和 deeplens_native_geometric；macOS GeoLens API IndexError 结构化处理（PSF smoke test 提前捕获，返回 GEOLENS_PSF_GEOMETRIC_FAILED_INDEXERROR）；ClaimGate 新增 proxy_as_native_geolens 违规类型（防止 proxy 实验被声明为 native GeoLens geometric PSF）；CLI 注册 run-deeplens-native-geolens-hsi-codesign 命令；remote command allowlist 新增对应条目。

Phase 38 增加 Agent selected plan 本地执行闭环：local mode 会跳过需要用户数据、远程执行或不支持后端的高分方案，执行最高分本地可执行 scientific design；若本地 scientific path 不可用，会结构化记录 unsupported 并尝试 report-only fallback。执行结果写入 ClaimGate、ResearchMemoryV2、StateStore snapshot、EventBus 和 plan execution report，report-only 不会升级 optical improvement claim。

Phase 39 补齐本地可执行 scientific handler：新增 `lightweight_scientific_execution` evidence level 和 `run_lightweight_mse_only_hsi()` 实验函数，使用 synthetic HSI + FFT PSF proxy 在无 DeepLens 环境下产出真实 metrics（MSE/PSNR/loss/improvement_detected）；`objective_redesign_simpler_metric_mse_only` design 由专属 handler 执行，优先于 report-only fallback；注册 `lightweight_scientific_hsi_mse_only` skill；ClaimGate 新增 `lightweight_as_native_physical` 和 `synthetic_metric_as_real_hsi` 违规类型防止声明升级。

Phase 40 新增 Handler Capability Registry 作为 handler 能力的唯一真相源，消除 design expected evidence 与 handler actual evidence 不一致的问题；`ExperimentDesignGenerator` 查询 registry 设置正确的 expected_evidence_level；`CandidatePlanEvaluator` 基于 actual_handler_evidence_level 评分并对 evidence downgrade 施加惩罚；ClaimGate 新增 `evidence_level_overestimated` 和 `handler_capability_exceeded` 违规；补齐 `param_reduction_sweep` local handler（第二个可执行 scientific path），sweep k=1,2,3 pseudo-optical parameters 并产出 lightweight metrics；`_build_attempt_sequence` 在 local mode 下优先选择 scientific handler。

Phase 41 新增 ClaimCeilingResolver，根据 handler capability / backend / dataset / execution fidelity 四级 ceiling 取最小值作为 final claim ceiling，彻底修复 design backend_id 导致 claim ceiling 过高的问题；ClaimGate 新增 ceiling 相关字段（final_claim_ceiling、ceiling_source、limiting_factor、downgrade_reasons）；execution result 携带完整 ceiling metadata；StateStore 记录 claim ceiling；Report 新增 Claim Ceiling Resolution 段；CLI 新增 resolve-claim-ceiling 命令。

Phase 42 将 HandlerCapabilityRegistry 从 hardcoded 迁移为 YAML 配置驱动（`optiresearch/config/handler_capabilities.yaml`），新增 schema validation（`handler_capability_schema.py`），5 个 enabled + 4 个 reserved disabled handler；registry 支持 reload / list_enabled / list_disabled / find_by_backend_id；ClaimCeilingResolver 处理 disabled handler；CLI 新增 validate-handler-capabilities / export-handler-capability-config-report；remote awareness 字段（supports_remote / remote_required / requires_remote_validation）预留给后续 remote execution。

Phase 43 激活 `remote_native_geolens_validation` handler（enabled），将 remote awareness 字段接入 Agent plan selection：CandidatePlanEvaluator 区分 local/remote_opt_in 模式下的 remote_required / supports_remote handler；AgentPlanExecutionLoop 新增 remote_opt_in 执行路径，通过 SkillRuntimeV2 调用 `run_remote_deeplens_native_geolens_hsi_codesign`；ClaimCeilingResolver 区分 local_evidence_ceiling 与 remote_evidence_ceiling；StateStore 新增 remote job / worker 状态追踪；Report 新增 Remote Evidence Ceiling table；remote handler 不允许绕过 allowlist。

Phase 44 完成 remote handler 端到端验收链路：`remote_native_geolens_validation` 在 `remote_opt_in + --allow-remote` 下优先选择；执行前校验 `RemoteWorkerRegistry` requirements、allowlist、runtime 与 artifact return path；SkillRuntimeV2 解析 `RemoteHandlerResult`，防止 fallback 或缺字段伪装 native；AgentPlanExecutionLoop 将 remote result 接入 ClaimGate、Memory、StateStore、EventBus 和 AgentPlanExecutionReport；CLI 补齐 `--allow-remote` 与 `--remote-worker-id`。

Phase 63 增加 component surrogate PSF HSI co-design：基于 Phase 62 已验证的 Fresnel / Binary2Phase component 参数语义，构造可微 surrogate PSF，接入 synthetic HSI forward 与 reconstruction loss，使 component parameters 能从 HSI loss 反传并更新。Claim ceiling 为 `component_surrogate_hsi_codesign`，不声明 full GeoLens lens-level optimization、real HSI performance 或 full wave-optics co-design。

Phase 68 建立 SystemCapabilityRegistry 与 ExecutionContract 统一层：新增 `SystemCapabilityEntry` / `SystemCapabilityRegistry` schema，从 7 个现有 registry 自动收集 capability 条目（handler / skill / design / backend / claim_policy），建立 `ExecutionContract` / `ArtifactContract` / `RemoteExecutionContract` / `ReportContract` 四种合约 schema，实现 4 个 validator 检查合约一致性、allowlist 覆盖、artifact 完整性、report 结构合规，生成 `ClaimPolicyMatrix`（16 evidence levels）与 `SystemCapabilityReport`（13 sections），输出 `ContractCoverageDashboard` 计算 `overall_system_readiness_score`。新增 8 个 CLI 命令与 23 个测试文件。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

可选绘图：

```bash
python -m pip install -e ".[dev,plot]"
```

## 快速开始

```bash
python -m optiresearch.cli init-db
python -m optiresearch.cli check-deeplens
python -m optiresearch.cli deeplens-capabilities
python -m optiresearch.cli run-mvp --backend mock_deeplens --objective "Design a mock depth-invariant and spectrally discriminative EDOF-HSI optical encoder"
python -m optiresearch.cli run-mvp --backend deeplens --objective "Design a minimal DeepLens PSF smoke run"
python -m optiresearch.cli query-memory --intent evidence --query "depth stability"
python -m optiresearch.cli run-benchmark --name opti-memory
python -m optiresearch.cli run-baselines --objective "Design depth-invariant and spectrally discriminative EDOF-HSI encoder"
python -m optiresearch.cli compare-backends --left mock_deeplens --right deeplens
python -m optiresearch.cli export-phase6-report
python -m optiresearch.cli export-paper-summary
python -m optiresearch.cli export-evidence-tables
python -m optiresearch.cli list-hsi-datasets
python -m optiresearch.cli prepare-hsi-dataset --dataset synthetic
python -m optiresearch.cli run-hsi-matrix --datasets synthetic --backends mock_deeplens --reconstructors optical_conditioned_linear,tiny_cnn --forward-modes depth_spectral_coded --objective "Compare encoder ranking across reconstructors"
python -m optiresearch.cli export-phase11-report
python -m optiresearch.cli run-public-hsi-matrix --dataset synthetic --backend mock_deeplens
python -m optiresearch.cli freeze-paper-protocol
python -m optiresearch.cli export-phase12-report
python -m optiresearch.cli add-remote-worker --worker-id windows_wsl --host wslbox --port 22 --username ysl --remote-project-dir /mnt/d/agent --remote-workspace-dir /mnt/d/agent/workspace --python-executable /mnt/d/agent/run_agent_python.sh
python -m optiresearch.cli check-remote-worker --worker-id windows_wsl
python -m optiresearch.cli run-remote-deeplens-source-smoke --worker-id windows_wsl
python -m optiresearch.cli run-agent-plan-execution --objective "recover from native GeoLens optical update instability" --seed-result-path workspace/native_geolens_stabilization/geolens_stabilization_1779550632/sweep_results.json --mode local --execute-top-k 1
python -m optiresearch.cli export-agent-plan-execution-report --execution-id <execution_id>
python -m optiresearch.cli run-component-surrogate-hsi-codesign --component fresnel --dataset synthetic --steps 3 --device cpu
python -m optiresearch.cli run-component-surrogate-hsi-codesign --component binary2phase --dataset synthetic --steps 3 --device cpu
python -m optiresearch.cli run-remote-component-surrogate-hsi-codesign --worker-id windows_wsl --component fresnel --dataset synthetic --steps 3 --device cpu
python -m optiresearch.cli export-component-surrogate-hsi-report --run-id <run_id>
python -m pytest
```

默认路径：

- `OPTIRESEARCH_DB_PATH=./workspace/optiresearch.sqlite`
- `OPTIRESEARCH_ARTIFACT_ROOT=./workspace/artifacts`

## CLI

```bash
python -m optiresearch.cli init-db
python -m optiresearch.cli check-deeplens
python -m optiresearch.cli deeplens-capabilities
python -m optiresearch.cli run-mvp --backend mock_deeplens --objective "Design a mock EDOF-HSI optical encoder"
python -m optiresearch.cli run-mvp --backend deeplens --objective "Design a minimal DeepLens PSF smoke run"
python -m optiresearch.cli query-memory --intent evidence --query "spectral separability"
python -m optiresearch.cli list-artifacts
python -m optiresearch.cli inspect-artifacts --run-id <run_id>
python -m optiresearch.cli explain-claim --claim-id <claim_id>
python -m optiresearch.cli list-plans
python -m optiresearch.cli match-plan --intent "evaluate edof hsi"
python -m optiresearch.cli list-skills-memory
python -m optiresearch.cli recommend-skills --intent "simulate psf"
python -m optiresearch.cli run-benchmark --name opti-memory
python -m optiresearch.cli run-benchmark --name opti-memory --mode full_rmos
python -m optiresearch.cli run-baselines --objective "Design depth-invariant and spectrally discriminative EDOF-HSI encoder"
python -m optiresearch.cli run-baselines --backend deeplens --objective "Design depth-invariant and spectrally discriminative EDOF-HSI encoder"
python -m optiresearch.cli compare-backends --left mock_deeplens --right deeplens
python -m optiresearch.cli export-phase6-report
python -m optiresearch.cli run-deeplens-smoke --objective "Design a minimal DeepLens PSF smoke run"
python -m optiresearch.cli explain-rule --rule-id <rule_id>
python -m optiresearch.cli export-paper-summary
python -m optiresearch.cli export-evidence-tables
python -m optiresearch.cli list-traces
python -m optiresearch.cli list-hsi-datasets
python -m optiresearch.cli prepare-hsi-dataset --dataset synthetic
python -m optiresearch.cli run-hsi-reconstruction --dataset synthetic --backend mock_deeplens --encoder controlled_chromatic_edof --forward-mode depth_spectral_coded --reconstructor optical_conditioned_linear --objective "Evaluate synthetic HSI reconstruction"
python -m optiresearch.cli run-hsi-matrix --datasets synthetic --backends mock_deeplens --reconstructors optical_conditioned_linear,tiny_cnn --forward-modes depth_spectral_coded --objective "Compare encoder ranking across reconstructors"
python -m optiresearch.cli export-phase11-report
python -m optiresearch.cli run-public-hsi-matrix --dataset synthetic --backend mock_deeplens
python -m optiresearch.cli freeze-paper-protocol
python -m optiresearch.cli export-phase12-report
python -m optiresearch.cli list-remote-workers
python -m optiresearch.cli check-remote-worker --worker-id windows_wsl
python -m optiresearch.cli run-remote-deeplens-source-smoke --worker-id windows_wsl
python -m optiresearch.cli run-remote-codesign --worker-id windows_wsl --objective "Run strict DeepLens-backed co-design on WSL D drive worker" --psf-source deeplens_parameterized --backend deeplens --fallback-policy fail --max-iterations 2
python -m optiresearch.cli export-remote-execution-report --job-id <job_id>
python -m optiresearch.cli run-agent-plan-execution --objective "recover from native GeoLens optical update instability" --seed-result-path workspace/native_geolens_stabilization/geolens_stabilization_1779550632/sweep_results.json --mode local --execute-top-k 1
python -m optiresearch.cli export-agent-plan-execution-report --execution-id <execution_id>
# Phase 24-27: Agentic Framework + Autonomous Loop + LLM Planner
python -m optiresearch.cli list-optical-backends
python -m optiresearch.cli inspect-optical-backend --backend-id deeplens_geolens_geometric
python -m optiresearch.cli run-experiment-v2 --objective "test" --backend-id phase_to_fft_proxy --task-type stable_lens_hsi_codesign --execution-target local
python -m optiresearch.cli recommend-next-strategy --latest-run-id <run_id> --backend-id deeplens_geolens_geometric
python -m optiresearch.cli compile-research-memory-v2
python -m optiresearch.cli query-research-memory-v2 --intent optimization_policy
python -m optiresearch.cli check-claim --claim-text "improves optimization" --backend-id phase_to_fft_proxy
python -m optiresearch.cli run-autonomous-research-loop-v2 --objective "test autonomous loop" --max-iterations 2 --execution-mode dry_run --planner-mode rule_based
python -m optiresearch.cli plan-with-llm --objective "investigate differentiable wave-optics alternatives" --provider mock
python -m optiresearch.cli list-planner-traces
python -m optiresearch.cli inspect-planner-trace --planner-run-id <id>
python -m optiresearch.cli export-llm-planner-report --planner-run-id <id>
# Phase 27: Real LLM validation
python -m optiresearch.cli check-llm-provider --provider deepseek
python -m optiresearch.cli export-llm-provider-validation-report --planner-run-id <id> --loop-id <id>
```

## API

```bash
uvicorn optiresearch.api.app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/v1/runs/mvp \
  -H "Content-Type: application/json" \
  -d '{"objective":"Design a mock EDOF-HSI optical encoder","workspace_id":"opti_lab"}'
```

## MVP Flow

1. `LeadInvestigator` 生成第一轮研究计划。
2. `MethodBuilder` 生成 mock optical spec 和 sweep spec。
3. `SimulationExperimentalist` 通过 allowlist 执行 `deeplens-adapter/run_mock_psf`。
4. `FileArtifactStore` 注册 PSF、MTF、metrics 和 manifest。
5. `MetaTraceWriter` 记录不可变事件。
6. `MemoryCompiler` 生成 `RunMemory`。
7. `CriticalReviewer` 生成并审查 simulation-only claim。
8. `MemoryRouter` 返回 evidence context pack。

## Phase 2

- `ExperimentSpec` 标准化 optical spec、sweep spec、metric spec 和 backend。
- `EvidenceEdge` 记录 artifact、metric、trace、score 和 rationale。
- `ArtifactInspector` 可读取 metrics JSON、NPZ、CSV 和图像文件摘要。
- `PlanTemplateManager` 从成功 run 编译可复用流程。
- `SkillMemoryManager` 从 trace 编译技能成功率、产物类型和适用偏好。
- `OptiMemoryBench` 运行 3 类 toy task，并输出 JSON / Markdown 报告。

## Phase 3

- `MockDeepLensAdapter` 支持 `conventional`、`achromatic`、`edof`、`chromatic_coded`、`controlled_chromatic_edof` 五类 encoder。
- `run-baselines` 自动运行 5 类 encoder 并输出 baseline comparison。
- `DesignRuleManager` 能从 claim / artifact metric 编译设计规则、检测反证、解释规则。
- `OptiMemoryBench` 支持 `no_memory`、`trace_only`、`plan_only`、`skill_only`、`full_rmos` 消融模式。

## Phase 4

- `ExperimentSpec`、`OpticalSpec`、`SweepSpec`、`MetricSpec` 固定 `schema_version="0.1"`。
- `validate_experiment_spec_version()` 检查 spec 版本漂移。
- `AdapterRunResult`、`AdapterArtifact`、`AdapterMetricBundle` 统一 mock 和真实 DeepLens 后端输出。
- `DeepLensAdapter` 在未安装真实 DeepLens 时返回结构化错误，不破坏本地测试和报告生成。
- `export-paper-summary` 输出 `workspace/reports/phase3_experiment_summary.md`。
- `export-evidence-tables` 输出 claim / rule evidence Markdown 表。

## Phase 5

- `check-deeplens` 输出真实 DeepLens 环境探测结果，包括 Python 版本、DeepLens 版本、import path 和 capabilities。
- `translate_experiment_spec()` 生成 `DeepLensCandidateConfig`，未支持字段保存在 `unsupported_fields`。
- `run-deeplens-smoke` 走真实 adapter；DeepLens 不可用时返回 `DEEPLENS_NOT_INSTALLED`。
- `run-mvp --backend deeplens` 在当前无真实 DeepLens 环境下会写 failed trace 和 failed RunMemory，claim 不会被标为 supported。
- `run-mvp --backend mock_deeplens` 保持原有 mock 闭环。

真实 DeepLens 来源：

- `https://github.com/vccimaging/DeepLens`
- Python 要求：DeepLens 当前项目元数据要求 `>=3.12`
- 安装示例：`python -m pip install "deeplens-core @ git+https://github.com/vccimaging/DeepLens.git"`

当前 adapter 已支持 `vccimaging/DeepLens` 的 `ParaxialLens.psf(points, ks=...)` 最小 PSF smoke 路径。

## Phase 6

- `.venv` 继续用于默认测试和 mock 后端。
- `.venv-deeplens` 用于真实 DeepLens 后端，当前安装 `deeplens-core==1.5.2` 和 Python `3.12.7`。
- `deeplens-capabilities` 输出 capability table，包括 import、ParaxialLens、PSF smoke、MTF export、encoder-specific design、optimization、HSI pipeline。
- `run-baselines --backend mock_deeplens` 输出 `workspace/baselines/mock_deeplens/`。
- `run-baselines --backend deeplens` 输出 `workspace/baselines/deeplens/`。
- `compare-backends` 输出 mock-real alignment JSON / Markdown。
- `export-phase6-report` 输出 `workspace/reports/phase6_real_deeplens_report.md`。
- 当前真实 DeepLens 后端是 smoke-level：验证 adapter、artifact、memory、evidence flow，不证明 encoder-specific optical behavior。

## Phase 7

- DeepLens 后端新增 encoder strategy registry，覆盖 `conventional`、`achromatic`、`edof`、`chromatic_coded`、`controlled_chromatic_edof`。
- 当前 Phase 7 是 real DeepLens base PSF generation + adapter-level encoder proxy transformation。
- 这不是 native physical encoder optimization，不能作为最终 optical performance claim。
- `run-baselines --backend deeplens` 现在会生成 encoder-specific PSF、MTF 和 metrics，并写入 `proxy_transform_manifest.json`。
- `compare-backends` 新增 proxy realization、native/proxy 区分、rank agreement 和 claims allowed / not allowed。
- `export-phase7-report` 输出 `workspace/reports/phase7_deeplens_encoder_proxy_report.md`。

Phase 7 常用命令：

```bash
MPLCONFIGDIR=workspace/mplconfig .venv-deeplens/bin/python -m optiresearch.cli deeplens-capabilities
MPLCONFIGDIR=workspace/mplconfig .venv-deeplens/bin/python -m optiresearch.cli run-baselines --backend deeplens --objective "Design depth-invariant and spectrally discriminative EDOF-HSI encoder"
python -m optiresearch.cli compare-backends --left mock_deeplens --right deeplens
python -m optiresearch.cli export-phase7-report
```

## Phase 8

- LLM provider layer is optional. Without API keys, agents keep rule-based fallback.
- `mock` provider is deterministic and used by tests.
- `deepseek` provider uses `https://api.deepseek.com/chat/completions` with `DEEPSEEK_API_KEY`.
- LLM output is schema-validated and cannot override artifact evidence.
- DeepLens now supports `semi_native` realization for the conventional ParaxialLens baseline when available.
- Other encoder families remain adapter-proxy unless experimental semi-native support is explicitly enabled and suitable API classes are detected.
- `OptimizationSpec` is draft-only; DeepLens optimization returns `OPTIMIZATION_NOT_AVAILABLE`.

Phase 8 commands:

```bash
python -m optiresearch.cli llm-providers
python -m optiresearch.cli check-llm
python -m optiresearch.cli test-llm --provider mock --prompt "Summarize OptiResearch Agent."
python -m optiresearch.cli run-mvp --use-llm --llm-provider mock --objective "Design a mock EDOF-HSI encoder"
python -m optiresearch.cli probe-deeplens-api
python -m optiresearch.cli export-phase8-report
```

## Phase 9

- Adds synthetic HSI dataset generation.
- Adds PSF-cube forward model for single-shot measurements.
- Adds numpy linear reconstruction baseline.
- Adds HSI metrics: PSNR, SSIM, SAM, ERGAS, per-band RMSE, worst-depth SAM.
- Adds reconstruction-level ClaimEvidence.
- Current results prove end-to-end evaluability, not final optical performance.

Commands:

```bash
python -m optiresearch.cli run-hsi-reconstruction --backend mock_deeplens --encoder controlled_chromatic_edof --objective "Evaluate synthetic HSI reconstruction with controlled chromatic EDOF encoder"
python -m optiresearch.cli run-hsi-baselines --backend mock_deeplens
python -m optiresearch.cli export-phase9-report
```

## Phase 11

- Adds `synthetic`, `local_npz`, `cave`, and `icvl` HSI dataset adapters.
- Public datasets are local-path only; no automatic download is performed.
- Adds optional Torch-based `tiny_cnn` and `unet_tiny` reconstructors. Default pytest does not require Torch.
- Adds `run-hsi-matrix` for dataset/backend/encoder/reconstructor/forward-mode comparison.
- Matrix ClaimEvidence distinguishes dataset, backend, reconstructor, and realization level.
- Synthetic/public dataset results with `mock_deeplens` are not real camera validation.
- DeepLens `adapter_proxy` is not native physical validation.

Commands:

```bash
python -m optiresearch.cli list-hsi-datasets
python -m optiresearch.cli prepare-hsi-dataset --dataset synthetic
python -m optiresearch.cli run-hsi-reconstruction --dataset synthetic --backend mock_deeplens --encoder controlled_chromatic_edof --forward-mode depth_spectral_coded --reconstructor optical_conditioned_linear --objective "Evaluate synthetic HSI reconstruction"
python -m optiresearch.cli run-hsi-matrix --datasets synthetic --backends mock_deeplens --reconstructors optical_conditioned_linear,tiny_cnn --forward-modes depth_spectral_coded --objective "Compare encoder ranking across reconstructors"
python -m optiresearch.cli export-phase11-report
```

## Phase 12

- `local_npz` now supports split NPZ files, `dataset.npz`, and single cube files.
- CAVE / ICVL adapters scan local `.npz`, `.npy`, and `.mat` files; no download is performed.
- DeepLens outputs now include wavelength-aware PSF contract metadata.
- Public HSI matrix writes structured skips when datasets or DeepLens are unavailable.
- ClaimEvidence distinguishes `public_hsi_mock`, `public_hsi_deeplens_proxy`, `public_hsi_deeplens_semi_native`, and `public_hsi_deeplens_native`.
- Paper experiment protocol v0.1 is frozen for reporting.

Commands:

```bash
python -m optiresearch.cli prepare-hsi-dataset --dataset local_npz --path /path/to/data --crop-size 32 --patch-stride 32 --normalization per_band
python -m optiresearch.cli run-public-hsi-matrix --dataset local_npz --path /path/to/data --backend mock_deeplens
python -m optiresearch.cli freeze-paper-protocol
python -m optiresearch.cli export-phase12-report
```

## Phase 13

Final benchmark freeze and paper evidence package.

- `python -m optiresearch.cli list-final-benchmarks` — List all 21 benchmark items
- `python -m optiresearch.cli collect-final-benchmark` — Export benchmark summary
- `python -m optiresearch.cli export-paper-tables` — Export 10 paper-ready tables (MD/CSV/JSON)
- `python -m optiresearch.cli export-claim-boundary` — Export claim whitelist/blacklist
- `python -m optiresearch.cli export-evidence-distribution` — Export evidence distribution
- `python -m optiresearch.cli export-warnings-audit` — Export warnings audit
- `python -m optiresearch.cli export-final-paper-package` — Export reproducibility package
- `python -m optiresearch.cli export-phase13-report` — Export Phase 13 report

See `docs/final_benchmark.md`, `docs/paper_tables.md`, `docs/claim_boundary.md`, `docs/final_paper_package.md`.

## Phase 18

DeepLens-backed black-box HSI co-design. DeepLens PSF enters the HSI forward
model and reconstruction chain via ParaxialLens. Black-box only:
`differentiable=false`, `native_parameter_update=false`.

## Phase 19

DeepLens Native Differentiable Optimization Probe. Systematically probes whether
DeepLens lens classes (ParaxialLens, GeoLens, DiffractiveLens, HybridLens,
PSFNetLens) support true gradient-based optical optimization with autograd chain:
`optical parameter → PSF → loss → backward → optimizer.step → parameter change`.

Commands:

```bash
# Inspect native optimization capabilities
python -m optiresearch.cli inspect-deeplens-native-optimization

# Run minimal native probe
python -m optiresearch.cli run-native-optimization-probe \
  --lens-class ParaxialLens --objective minimize_psf_width --max-steps 2 --device cpu

# Run remotely on WSL
python -m optiresearch.cli run-remote-native-optimization-probe \
  --worker-id windows_wsl --lens-class ParaxialLens \
  --objective minimize_psf_width --max-steps 2 --device cpu

# Export report
python -m optiresearch.cli export-phase19-report
```

See `docs/deeplens_native_optimization_probe.md`, `docs/native_optimization_claims.md`.

## Phase 19B

Correct DeepLens native optimization path discovery. Phase 19B no longer treats
`ParaxialLens` as the only native optimization evidence. It scans DeepLens
source paths and probes diffractive/phase surfaces first, then tries lens-file
classes separately.

Commands:

```bash
python -m optiresearch.cli scan-deeplens-optimization-paths

python -m optiresearch.cli run-deeplens-surface-optimization-probe \
  --surface Fresnel --objective minimize_phase_variance --max-steps 3

python -m optiresearch.cli run-deeplens-surface-optimization-probe \
  --surface Binary2Phase --objective match_target_phase --max-steps 3

python -m optiresearch.cli run-deeplens-lensfile-optimization-probe \
  --lens-class GeoLens --max-files 5 --max-steps 2

python -m optiresearch.cli run-remote-deeplens-surface-optimization-probe \
  --worker-id windows_wsl --surface Fresnel \
  --objective minimize_phase_variance --max-steps 3

python -m optiresearch.cli export-phase19b-report
```

Claim boundary:
- Surface probe success supports component-level native differentiable optimization.
- Lens-level native optimization requires lens-file load + PSF/image loss backward + parameter change.
- Native optical-HSI co-design still requires HSI loss to reach an optical parameter.

## Phase 58-59

Remote GeoLens diagnostics with cross-platform lens file resolution.

Commands:

```bash
# Resolve lens file path
python -m optiresearch.cli resolve-lens-file \
  --lens-file auto:cooke --backend-id deeplens_geolens_geometric

# Remote lens resolution
python -m optiresearch.cli run-remote-resolve-lens-file \
  --worker-id windows_wsl --lens-file auto:cooke

# Remote trainable parameter inspection
python -m optiresearch.cli run-remote-deeplens-trainable-parameter-inspection \
  --worker-id windows_wsl --lens-file auto:cooke --device cpu

# Remote autograd audit
python -m optiresearch.cli run-remote-deeplens-autograd-audit \
  --worker-id windows_wsl --lens-file auto:cooke --device cpu

# Remote curriculum probe
python -m optiresearch.cli run-remote-deeplens-curriculum-probe \
  --worker-id windows_wsl --max-steps 3 --device cpu

# Remote regularized probe
python -m optiresearch.cli run-remote-deeplens-regularized-probe \
  --worker-id windows_wsl --max-steps 3 --device cpu

# Export diagnostic report
python -m optiresearch.cli export-remote-diagnostic-report \
  --remote-job-id <remote_job_id>
```

See `docs/lens_file_resolver.md`, `docs/wsl_lens_file_resolution.md`, `docs/remote_geolens_diagnostics.md`.

Claim boundary:
- All remote diagnostics capped at `diagnostic_evidence`.
- No optical improvement claims from diagnostic data.
- Lens file resolution is infrastructure — no scientific claims.

## Phase 60-61

GeoLens autograd validation on WSL and component-level recovery pivot.

Key findings:
- The original WSL finding used `geolens.parameters()`, which is not valid for
  DeepLens `GeoLens`
- The corrected route uses `get_optimizer_params()` / `get_optimizer()` and
  float32 geometric PSF
- `full_geolens_direct_update` is **conditional**: allowed only after native
  optimizer audit proves connected gradients and a parameter update
- Pivot strategy: validate individual component backends (Fresnel, Binary2Phase)

Commands:
```bash
# Remote trainable parameter inspection
python -m optiresearch.cli run-remote-deeplens-trainable-parameter-inspection \
  --worker-id windows_wsl

# Remote autograd audit
python -m optiresearch.cli run-remote-deeplens-autograd-audit \
  --worker-id windows_wsl

# Agent plan execution with diagnosis
python -m optiresearch.cli run-agent-plan-execution \
  --objective "validate GeoLens autograd" \
  --mode remote_opt_in --use-gradient-diagnosis \
  --allow-remote --remote-worker-id windows_wsl --execute-top-k 2
```

See `docs/non_differentiable_geolens_path_policy.md`, `docs/component_level_pivot.md`.

## Phase 62

Validate DeepLens component backends (Fresnel, Binary2Phase, diffractive candidates)
on WSL. Component-level probes confirm trainability and differentiability of
individual surface components.

Commands:
```bash
# Component discovery
python -m optiresearch.cli discover-deeplens-components
python -m optiresearch.cli run-remote-discover-deeplens-components \
  --worker-id windows_wsl

# Component probes
python -m optiresearch.cli run-deeplens-component-probe \
  --component fresnel --device cpu
python -m optiresearch.cli run-remote-deeplens-component-probe \
  --worker-id windows_wsl --component fresnel --device cpu

# Report
python -m optiresearch.cli export-component-probe-report \
  --remote-job-id <remote_job_id>
```

Claim boundaries:
- Component probes capped at `native_component_optimization`.
- Component evidence alone does not support `full_geolens_direct_update`.
- No HSI improvement claims from component evidence.
- No lens-level optimization claims from component evidence.

See `docs/component_backend_discovery.md`, `docs/fresnel_component_probe.md`,
`docs/binary2phase_component_probe.md`, `docs/component_probe_claim_boundaries.md`,
`docs/wsl_component_backend_validation.md`.
