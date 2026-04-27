# SFA V1 O 矩阵切分分析报告

> 分析范围：vllm-ascend `AscendSFAImpl`（`attention/sfa_v1.py`），聚焦 DeepSeek V3.2 模型。
>
> 分析日期：2026-04-27

---

## 1. DeepSeek V3.2 MLA 关键参数

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| `kv_lora_rank` | L | 512 | KV 压缩隐层维度 |
| `qk_nope_head_dim` | P | 128 | Q/K 不走 RoPE 的维度 |
| `qk_rope_head_dim` | R | 64 | Q/K 走 RoPE 的维度 |
| `v_head_dim` | V | 128 | V head 维度 |
| `qk_head_dim` | P+R | 192 | QK head 总维度 |
| `num_attention_heads` | N_total | 128 | 全局 attention head 数 |
| `hidden_size` | H | 7168 | 模型隐层维度 |
| `q_lora_rank` | Lq | 1536 | Q 压缩隐层维度 |
| Indexer `n_head` | - | 64 | Indexer 稀疏选择的 head 数 |
| Indexer `head_dim` | - | 128 | Indexer head 维度 |

**TP 相关衍生量**（以 `tp` 表示 TP 并行度）：

| 参数 | 公式 | 示例（tp=4） |
|------|------|-------------|
| `num_heads`（传给 impl 的每卡 head 数） | N_total / tp | 32 |
| `local_num_heads`（非 DSA-CP） | = num_heads | 32 |
| `local_num_heads`（DSA-CP） | num_heads × tp = N_total | 128 |
| W_UV 形状（非 DSA-CP） | (N_total/tp, L, V) | (32, 512, 128) |
| W_UV 形状（DSA-CP） | (N_total, L, V) | (128, 512, 128) |

> **关键代码路径**：`num_heads` 的值链为 `config.num_attention_heads(128)` → `num_local_heads(128/tp)` → `MultiHeadLatentAttentionWrapper` → `MLAAttention` → `AscendSFAImpl`。因此 impl 收到的 `num_heads` 是**每卡值**。
>
> **参考文件**：`vllm/model_executor/models/deepseek_v2.py:953`、`vllm/model_executor/layers/mla.py:73`、`vllm/attention/layer.py:570`。

---

## 2. O 矩阵通用计算流程

O 矩阵的计算经过三个阶段：**Sparse Flash Attention** → **V Up Projection** → **O Projection**。

### 2.1 权重初始化（`process_weights_after_loading`）

```text
kv_b_proj.weight (ColumnParallelLinear, 已 TP 切分)
  shape: (N_per_rank * (P+V), L)  -- 例: (8192, 512) @ tp=4
  .T → (L, N_per_rank * (P+V))
  view → (L, N_per_rank, P+V)
  split → W_UK (L, N_per_rank, P)  +  W_UV (L, N_per_rank, V)
  transpose/permute:
    W_UV  → (N_per_rank, L, V)     -- V 升维权重
    W_UK_T → (N_per_rank, P, L)    -- K 升维权重（已转置）
  dispose kv_b_proj (节省显存)
```

**DSA-CP 模式下的差异**：

- `kv_b_proj` 使用 `ShardedCPColumnParallelOp`（fake comm_group, world_size=1），绕过 TP 切分
- 每卡加载**全量** kv_b_proj 权重：`(N_total * 256, L)` = `(32768, 512)`
- W_UV → `(N_total, L, V)` = `(128, 512, 128)`
- `local_num_heads = N_total = 128`

> **参考代码**：`ops/linear_op.py:608-631`（ShardedCPColumnParallelOp）、`sfa_v1.py:467-468`

### 2.2 Sparse Flash Attention 输出

```text
query  = ql_nope: (T, N, L)    -- Q 经 W_UK 升维到 latent 空间
query_rope = q_pe: (T, N, R)   -- Q 的 RoPE 部分
key    = k_nope_cache (latent KV, 每个序列位置 1 个 head)
value  = k_nope_cache (MLA 中 K=V)
key_rope = k_pe_cache

→ attn_output: (T, N, L)       -- latent 空间的 attention 加权和
```

其中 T 取决于模式：标准 TP 时 T = 总 token 数；DSA-CP 时 T = T_local（每卡分配到的 token 子集）。

### 2.3 V Up Projection（`_v_up_proj`）

```text
attn_output: (T, N, L) = (T, N, 512)
  bmm × W_UV (N, L, V) = (N, 512, 128)
  = attn_output_v: (T, N, V) = (T, N, 128)
  flatten → (T, N*V)
```

| 模式 | N | 输出形状 |
|------|---|---------|
| 标准 TP | N_per_rank = 32 (tp=4) | (T, 4096) |
| DSA-CP | N_total = 128 | (T, 16384) |

### 2.4 O Projection

O projection 将 V 升维后的结果映射回 hidden_size：

```text
attn_output_v: (T, N*V)
  × o_proj weight
  = output: (T, H) = (T, 7168)
```

O projection 的具体实现方式是**各种模式间最大的差异点**，详见后续各节。

---

## 3. PD 分离模式（Disaggregated）

### 3.1 模式判定条件

PD 分离模式的四个判定条件，按依赖关系从底层到顶层排列：

| 条件 | P 节点 | D 节点 | 判定逻辑 |
|------|--------|--------|---------|
| `kv_transfer_config.kv_role` | `"kv_producer"` | `"kv_consumer"` | 启动参数指定 |
| `enable_dsa_cp` | True（需启用 FlashComm1） | **False**（FlashComm1 通常不启用） | 模型类型 + FlashComm1/DynamicEPLB |
| `enable_dsa_cp_with_layer_shard` | True | False | DSA-CP 开启 **且** `kv_role == "kv_producer"` |
| `enable_dsa_cp_with_o_proj_tp` | False | False | DSA-CP 开启 **且** `kv_role in (None, "kv_both")` |

**条件 ① — `kv_role`**（`vllm/vllm/config/kv_transfer.py`）：

| 判定位 | 行号 | 说明 |
|------|------|------|
| `KVProducer` / `KVConsumer` 类型 | `L13-L15` | `"kv_producer"` / `"kv_both"` 和 `"kv_consumer"` / `"kv_both"` |
| `kv_role` 字段定义 | `L38-L40` | 用户通过 CLI `--kv-transfer-config` 指定 |
| `is_kv_producer` property | `L111-L112` | `kv_connector is not None and kv_role in KVProducer` |
| `is_kv_consumer` property | `L115-L116` | `kv_connector is not None and kv_role in KVConsumer` |

> P 节点和 D 节点区分的关键在于 `KVProducer = ("kv_producer", "kv_both")` 与 `KVConsumer = ("kv_consumer", "kv_both")` 的取反关系。
>
> P 节点（`is_kv_producer=True, is_kv_consumer=False`）之所以 `is_kv_consumer=False`，是因为 `kv_role="kv_producer"` 不在 `KVConsumer` 的合法值中。同理 D 节点反之。

**条件 ② — `enable_dsa_cp`**（`vllm-ascend/vllm_ascend/utils.py:1271-1278`）：

```python
# utils.py:1271-1278
def enable_dsa_cp() -> bool:
    vllm_config = get_current_vllm_config()
    is_ds_v32 = hasattr(vllm_config.model_config, "hf_text_config") and hasattr(
        vllm_config.model_config.hf_text_config, "index_topk"
    )
    return bool(is_ds_v32 and enable_sp())
```

| 子条件 | 判定逻辑 | 代码位置 |
|------|---------|---------|
| `is_ds_v32` | 模型 config 中是否存在 `index_topk` 属性（DeepSeek V3.2 特有） | `utils.py:1275-L1277` |
| `enable_sp()` | `VLLM_ASCEND_ENABLE_FLASHCOMM1` 环境变量 或 `VLLM_ASCEND_ENABLE_FLASHCOMM`（向后兼容） | `utils.py:799-817` 或 Dynamic EPLB 自动启用 |

> **D 节点不启用 DSA-CP 的原因**：D 节点通常不设置 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`（FlashComm1 的 KV all-gather 只在 prefill 阶段有意义，decode 阶段仅需少量 token，all-gather 得不偿失）。因此 `enable_sp()` 返回 False，`enable_dsa_cp()` 也返回 False，D 节点回退到标准 MLA TP 路径。

**条件 ③ — `enable_dsa_cp_with_layer_shard`**（`vllm-ascend/vllm_ascend/utils.py:1282-1292`）：

```python
# utils.py:1282-1292
def enable_dsa_cp_with_layer_shard() -> bool:
    if not enable_dsa_cp():
        return False
    vllm_config = get_current_vllm_config()
    kv_transfer_config = vllm_config.kv_transfer_config
    is_prefill_instance = (
        kv_transfer_config is not None
        and kv_transfer_config.kv_role == "kv_producer"
    )
    return is_prefill_instance
```

> Layer Sharding 的 payload（广播 o_proj 权重）只在 prefill 计算密集阶段能被计算掩盖，因此**只对纯 P 节点（`kv_producer`）启用**。`kv_both`（混部）也不会走这条路径。

**条件 ④ — `enable_dsa_cp_with_o_proj_tp`**（`vllm-ascend/vllm_ascend/utils.py:1296-1307`）：

```python
# utils.py:1296-1307
def enable_dsa_cp_with_o_proj_tp() -> bool:
    if not enable_dsa_cp():
        return False
    vllm_config = get_current_vllm_config()
    kv_transfer_config = vllm_config.kv_transfer_config
    return kv_transfer_config is None or kv_transfer_config.kv_role == "kv_both"
```

> 当 `kv_role` 是 `"kv_producer"` 或 `"kv_consumer"`（PD 分离侧），此开关**必然为 False**。只有无 KV transfer 或 `"kv_both"`（混部）才启用 o_proj TP 模式。这就是 PD 分离 P/D 节点都不走 all-to-all decode 路径的根本原因。

**在 `AscendSFAImpl.__init__` 中的使用**（`vllm-ascend/vllm_ascend/attention/sfa_v1.py:458-465`）：

```python
# sfa_v1.py:458-465
self.enable_dsa_cp = enable_dsa_cp()
self.enable_dsa_cp_with_layer_shard = enable_dsa_cp_with_layer_shard()
self.enable_dsa_cp_with_o_proj_tp = enable_dsa_cp_with_o_proj_tp()
```

这三个布尔值在 `forward()` 中控制 KV all-gather 策略（`sfa_v1.py:1133-1143`）、O projection 路径选择（`sfa_v1.py:1260`）、以及 layer sharding 权重管理（`sfa_v1.py:467-479`）。

### 3.2 P 节点（Prefill Only）— O 矩阵切分

**核心特征**：DSA-CP 模式 + Layer Sharding

```text
+-------------------------------------------------+
| P 节点 (kv_producer)                             |
|                                                  |
| Token 切分: T_local = ceil(T_total / tp)         |
|                                                  |
| 1. KV All-Gather (async):                        |
|    各卡的 k_pe, k_nope, k_li 拼接后 all_gather    |
|    → 全量 KV (T_total 个 token)                   |
|    → 写回 KV cache (共享给全卡)                    |
|                                                  |
| 2. Q Projection:                                 |
|    q_c: (T_local, 1536)                          |
|    q_proj → q_nope: (T_local, 128, 128)          |
|    bmm × W_UK_T → ql_nope: (T_local, 128, 512)  |
|                                                  |
| 3. Sparse Flash Attention:                       |
|    attn_output: (T_local, 128, 512)              |
|                                                  |
| 4. V Up Projection:                              |
|    W_UV: (128, 512, 128)                         |
|    attn_output_v: (T_local, 128*128)=(T_local,16384) |
|                                                  |
| 5. O Projection (Layer Sharding):                |
|    o_proj 权重在设备间轮流存储（每卡只存部分层的权重）|
|    计算时 async broadcast 权重到本卡               |
|    o_proj: (T_local, 16384) → (T_local, 7168)    |
|    不需要 all-reduce（每卡处理不同的 token）       |
+-------------------------------------------------+
```

**代码位置索引**（按流程顺序）：

| 步骤 | 说明 | 文件 | 关键行号 |
|------|------|------|---------|
| Token 切分 | `T_local = ceil(T / tp)`，padding 对齐、slot_mapping 切分、cos/sin 切分 | `attention/sfa_v1.py` | `L256-323` |
| Q Projection | `_q_proj_and_k_up_proj`：q_proj → split nope/rope → bmm × W_UK_T | `attention/sfa_v1.py` | `L816-828` |
| KV 处理 + All-Gather | `exec_kv`（DSA-CP 模式，输出 k_pe/k_nope 不写 cache）→ 拼接 `fused_kv` → `all_gather_async` | `attention/sfa_v1.py` | `L787-800`（exec_kv）、`L1144-1157`（all_gather） |
| KV Cache 回写 | wait all-gather → 拆分 k_pe/k_nope/k_li → `reshape_and_cache` 写全量 slot_mapping | `attention/sfa_v1.py` | `L1199-1232` |
| Sparse Flash Attention | `_execute_sparse_flash_attention_process`：`npu_sparse_flash_attention` 算子 | `attention/sfa_v1.py` | `L1030-1053` |
| V Up Projection | `_v_up_proj`：`batch_matmul_transpose`（fast path）或 bmm × W_UV | `attention/sfa_v1.py` | `L830-848` |
| O Projection (Layer Sharding) | `self.o_proj(attn_output)`，内部走 `ShardedCPRowParallelOp` | `attention/sfa_v1.py` | `L1278` |
| KV 保存到 Connector | `maybe_save_kv_layer_to_connector`（P 节点特产，供 D 节点加载） | `attention/sfa_v1.py` | `L1280` |
| reshape_cache_event | `is_kv_producer` 时记录 event，标记 KV 写入完成 | `attention/sfa_v1.py` | `L1233-1234` |

**Token 切分细节**（`sfa_v1.py:256-323`）：

> 总 token 数 `num_tokens` 向上 pad 到 `tp_size` 的倍数，每卡取 `[rank * chunk, (rank+1) * chunk)` 这一段。同时切分 `slot_mapping_cp`（本卡写 KV 用的 local slot）、`cos`/`sin`（本卡的 RoPE 参数）、`actual_seq_lengths_query/key`（本卡视角的 cum lengths）。

**KV All-Gather 异步控制**（`sfa_v1.py:1143`）：

> `async_op = self.enable_dsa_cp_with_layer_shard or full_gather_o_proj_enabled` — P 节点 layer sharding 模式下 `async_op=True`，KV all-gather 与后续计算重叠。

**Layer Sharding 机制详解**：

| 子步骤 | 文件 | 关键行号 |
|------|------|---------|
| 注册层到分片系列 | `ops/layer_shard_linear.py` | `register_all_layers_to_shard_weight_series` `L279-290` |
| 单层注册（覆盖 forward，dispose 非本卡权重） | `ops/layer_shard_linear.py` | `register_layer_to_shard_weight_series` `L205-247` |
| init 中收集 layer_sharding kwargs 并注册 | `attention/sfa_v1.py` | `L473-483` |
| forward 中触发 async broadcast | `attention/sfa_v1.py` | `L1190-1197`（`reach_layer_for_shard_weight_series`） |
| `ShardedCPRowParallelOp`：fake comm（world_size=1），绕过 all-reduce | `ops/linear_op.py` | `L585-606` |

**Layer Sharding 要点**：

- 层权重按 `layer_idx % world_size` 分散存储，非本卡层数 dispose 掉不做 weight loading
- `reach_layer_for_shard_weight_series` 触发 async broadcast 拉取当前层的权重
- `ShardedCPRowParallelOp.comm_group` 返回 fake group（world_size=1），让上层 RowParallelLinear 认为无需 all-reduce
- `prefetch_step=1`：提前 1 层发起 broadcast，实现通信计算重叠

### 3.3 D 节点（Decode Only）— O 矩阵切分

**核心特征**：标准 MLA TP（非 DSA-CP）

```text
+-------------------------------------------------+
| D 节点 (kv_consumer)                             |
|                                                  |
| 标准 TP: 每卡处理 T 个 token, 各自负责 N_per_rank 个 head |
|                                                  |
| 1. KV 从 connector 加载（由 P 节点 transfer 来）  |
|                                                  |
| 2. Sparse Flash Attention:                       |
|    attn_output: (T, N_per_rank, 512)             |
|    例: (T, 32, 512) @ tp=4                       |
|                                                  |
| 3. V Up Projection:                              |
|    W_UV: (32, 512, 128)                          |
|    attn_output_v: (T, 32*128) = (T, 4096)        |
|                                                  |
| 4. O Projection (标准 RowParallelLinear):         |
|    o_proj: (T, 4096) → partial (T, 7168)         |
|    all-reduce → output (T, 7168)                 |
+-------------------------------------------------+
```

**O 矩阵切分方式**：标准 TP 行切分

- o_proj 是 `RowParallelLinear`，输入维度按 TP 切分
- 每卡处理 N_per_rank 个 head 对应的 V 输出（共 N_per_rank × V = 4096 维）
- 每卡输出 partial result，all-reduce 后得到完整 hidden states

---

## 4. PD 混部模式（Colocation）

### 4.1 模式判定条件

| 条件 | 值 |
|------|-----|
| `kv_transfer_config.kv_role` | `"kv_both"` |
| `enable_dsa_cp` | True |
| `enable_dsa_cp_with_layer_shard` | False |
| `enable_dsa_cp_with_o_proj_tp` | True |

### 4.2 Prefill 阶段 — O 矩阵切分

**核心特征**：DSA-CP + o_proj 全量权重切换

```text
+-------------------------------------------------+
| PD 混部 - Prefill 阶段                           |
|                                                  |
| Token 切分: T_local = ceil(T_total / tp)         |
| N = N_total = 128 (全量头)                        |
|                                                  |
| 1. KV All-Gather (async):                        |
|    各卡 KV all_gather → 全量 KV                   |
|    o_proj TP 权重并行 all_gather → full_pool      |
|                                                  |
| 2. Sparse Flash Attention:                       |
|    attn_output: (T_local, 128, 512)              |
|                                                  |
| 3. V Up Projection:                              |
|    attn_output_v: (T_local, 128*128)             |
|                  = (T_local, 16384)               |
|                                                  |
| 4. O Projection (全量权重模式):                    |
|    ① wait o_proj_full_handle                     |
|    ② set o_proj.weight → o_proj_full_pool (全量)  |
|    ③ o_proj: (T_local, 16384) → (T_local, 7168)  |
|    ④ set o_proj.weight → o_proj_tp_weight (切回)  |
|                                                  |
|    每卡独立计算全量结果，无需 all-reduce             |
+-------------------------------------------------+
```

**权重切换机制**（`_handle_o_proj_weight_switch_and_forward`）：

- 预分配 `o_proj_full_pool`：shape 为 `(tp * out_dim_per_rank, in_dim_per_rank)`
- Prefill 开始前，异步 all-gather 所有 rank 的 o_proj 权重到 full_pool
- 计算时临时切换到全量权重，执行 matmul，然后切回 TP 权重
- 量化参数（`aclnn_input_scale` 等）也做相应切换

> **参考代码**：`sfa_v1.py:686-716`（init）、`sfa_v1.py:718-762`（forward）

PD  混部的 prefill 阶段，实际上就是一个经典的 CP 方案的 O 矩阵过程，AllGather 一下全量的权重，然后每卡上是按 token 分片的激活值，算出来之后直接进入下一层的 CP 版本的 Attention。

### 4.3 Decode 阶段（All-to-All）— O 矩阵切分（详细分析）

**核心特征**：DSA-CP + Activation All-to-All

以下从 `forward()` 入口出发，逐步追踪 tensor shape。

---

#### Step 1: 入口与 DSA-CP 元数据准备

```python
# sfa_v1.py forward(), 行 1055-1084
# 输入
hidden_states: (num_input_tokens, H) = (num_input_tokens, 7168)
kv_cache = [k_nope_cache, k_pe_cache, k_li_cache]

# DSA-CP 元数据
dsa_cp_context.slot_mapping_cp  # 本卡 token 对应的 slot
dsa_cp_context.actual_seq_lengths_query  # 本卡视角的 cumulative query lengths
dsa_cp_context.actual_seq_lengths_key    # 本卡视角的 sequence lengths

# 判定为 decode 阶段
full_gather_o_proj_enabled = False  # decode 不走权重 gather
```

---

#### Step 2: Fused QKV A Projection

```python
# sfa_v1.py 行 1113-1121
qkv_lora = fused_qkv_a_proj(hidden_states)[0]
# shape: (T_local, q_lora_rank + kv_lora_rank + qk_rope_head_dim)
#       = (T_local, 1536 + 512 + 64)
#       = (T_local, 2112)

q_c, kv_no_split = qkv_lora.split([1536, 576], dim=-1)
# q_c:       (T_local, 1536)   -- Q 压缩隐表示
# kv_no_split: (T_local, 576)  -- KV 压缩隐表示 (L+R)
```

同时，Indexer 预处理也在本步完成：

```python
# sfa_v1.py 行 1126
k_li, k_li_scale = indexer_select_pre_process(x=hidden_states, cos=cos, sin=sin)
# k_li: (T_local, 1, 128)  -- Indexer 的 latent key (n_head=64, head_dim=128)
```

---

#### Step 3: KV 处理 + All-Gather

```python
# sfa_v1.py 行 1130-1134
k_pe, k_nope = exec_kv(kv_no_split, cos, sin, kv_cache, slot_mapping_cp, ...)
# k_pe:   (T_local, 1, 64)   -- RoPE 部分
# k_nope: (T_local, 1, 512)  -- Latent KV 部分
```

**KV All-Gather**（关键通信操作）：

```python
# sfa_v1.py 行 1143-1154
fused_kv = cat([
    k_pe.view(-1, 64),     # (T_local, 64)
    k_nope.view(-1, 512),  # (T_local, 512)
    k_li.view(-1, 128),    # (T_local, 128)
], dim=1)
# fused_kv: (T_local, 704)

fused_kv_no_split, kv_ag_handle = all_gather_async(fused_kv, tp_group, async_op=True)
# fused_kv_no_split: (T_local * tp, 704) = (T_total, 704)
#   ↑ 包含所有 rank 的 KV，按 token 拼接
```

**等待 all-gather 完成并拆分**：

```python
# sfa_v1.py 行 1184-1205
kv_ag_handle.wait()

k_pe, k_nope, k_li = fused_kv_no_split.split([64, 512, 128], dim=-1)
# k_pe:   (T_total, 64)
# k_nope: (T_total, 512)
# k_li:   (T_total, 128)

# 写回全量 KV 到 cache（使用全量 slot_mapping）
k_nope = k_nope.view(T_total, 1, 512)
k_pe = k_pe.view(T_total, 1, 64)
reshape_and_cache(
    key=k_nope[:num_actual_tokens],     # (T_actual, 1, 512)
    value=k_pe[:num_actual_tokens],      # (T_actual, 1, 64)
    key_cache=kv_cache[0],
    value_cache=kv_cache[1],
    slot_mapping=slot_mapping[:num_actual_tokens],  # 全量 slot
)
```

> **注意**：写 cache 用的是**全量 slot_mapping**（非 slot_mapping_cp），因为 cache 需要所有 token 的 KV。

---

#### Step 4: Q Projection + K Up Projection

```python
# sfa_v1.py 行 1180-1181, _q_proj_and_k_up_proj
ql_nope, q_pe = _q_proj_and_k_up_proj(q_c)

# 内部过程:
# q_proj(q_c) → (T_local, 128 * 192) = (T_local, 24576)
# view → (T_local, 128, 192)
# split → q_nope: (T_local, 128, 128), q_pe_raw: (T_local, 128, 64)

# K Up Projection (Q × W_UK_T):
# q_nope.T → (128, T_local, 128)
# bmm × W_UK_T (128, 128, 512) → (128, T_local, 512)
# .T → ql_nope: (T_local, 128, 512)

# RoPE:
q_pe = rope_single(q_pe_raw, cos, sin)
# q_pe: (T_local, 128, 64)
```

---

#### Step 5: Indexer + Sparse Flash Attention

```python
# sfa_v1.py 行 1233-1246

# Indexer 后处理 → topk_indices
topk_indices = indexer_select_post_process(...)
# topk_indices: 稀疏选择的 token 索引

# Sparse Flash Attention
attn_output = npu_sparse_flash_attention(
    query=ql_nope,          # (T_local, 128, 512)  -- latent 空间
    key=kv_cache[0],        # k_nope cache (latent KV)
    value=kv_cache[0],      # MLA: value = key
    query_rope=q_pe,        # (T_local, 128, 64)
    key_rope=kv_cache[1],   # k_pe cache
    sparse_indices=topk_indices,
    ...
)
# attn_output: (T_local, 128, 512)
#   ↑ 在 latent 空间的 attention 加权和
#   第 1 维是本卡分配的 token 数
#   第 2 维是全量 128 个 head
#   第 3 维是 latent 维度 L=512
```

---

#### Step 6: V Up Projection

```python
# sfa_v1.py 行 1248
attn_output = _v_up_proj(attn_output)

# 内部（fast path, batch_matmul_transpose）:
# attn_output: (T_local, 128, 512)
# W_UV: (128, 512, 128)
# res = bmm_transpose(attn_output, W_UV)
#   → (T_local, 128, 128)
# flatten → (T_local, 128 * 128) = (T_local, 16384)
```

**Shape 变化总结**：

```text
(T_local, 128, 512) × (128, 512, 128) → (T_local, 128, 128) → (T_local, 16384)
   attn_output        W_UV                  attn_output_v       flattened
```

---

#### Step 7: All-to-All（Decode 特有的关键操作）

```python
# sfa_v1.py 行 751-762, _handle_o_proj_weight_switch_and_forward (decode 分支)
# should_shard_weight = False (decode), 进入 else 分支

# 输入: attn_output: (T_local, 16384)

send = (
    attn_output
    .view(-1, self.tp_size, self.num_heads * self.v_head_dim)
    # → (T_local, tp, N_per_rank * V) = (T_local, 4, 32*128) = (T_local, 4, 4096)
    .permute(1, 0, 2)
    # → (tp, T_local, 4096) = (4, T_local, 4096)
    .reshape(-1, self.num_heads * self.v_head_dim)
    # → (tp * T_local, 4096) = (4 * T_local, 4096)
)

attn_output = torch.empty_like(send)
# → (4 * T_local, 4096)  预分配输出

torch.distributed.all_to_all_single(attn_output, send, group=tp_group)
# 每个 rank 发送 tp 个 chunk，每个 chunk (T_local, 4096)
# 每个 rank 接收 tp 个 chunk，每个 chunk (T_local, 4096)
# → attn_output: (tp * T_local, 4096) = (T_total, 4096)
```

**All-to-All 的语义**：

```text
Before all-to-all (每卡视角):
  shape: (tp * T_local, 4096)
  逻辑分块: tp 个 chunk，chunk[i] 包含 rank i 负责的 head 对应的 V 输出
  每个 chunk 有 T_local 个 token

After all-to-all (每卡视角):
  shape: (tp * T_local, 4096)
  逻辑分块: tp 个 chunk，chunk[i] 来自 rank i 的 token
  每个 chunk 有 T_local 个 token，都是本卡负责的 head

总结:
  before: T_local tokens × all heads (split into tp head-chunks)
  after:  T_total tokens × own heads (N_per_rank heads)
```

---

#### Step 8: O Projection（标准 TP）

```python
# sfa_v1.py 行 1271
output[...] = self.o_proj(attn_output)[0]

# attn_output: (T_total, N_per_rank * V) = (T_total, 4096)  with tp=4

# o_proj 是 RowParallelLinear:
#   weight: (N_per_rank * V, H) per rank  = (4096, 7168) with tp=4
#   (或等效的 Ascend 量化格式)
#   forward: partial = matmul(attn_output, weight)  → (T_total, 7168)
#            output = all_reduce(partial)             → (T_total, 7168)
```

---

#### Decode 全流程 Shape 总结

```text
hidden_states        (num_input_tokens, 7168)
    |
    v [fused_qkv_a_proj]
qkv_lora             (T_local, 2112)
    +-- q_c          (T_local, 1536)
    +-- kv_no_split  (T_local, 576)
          |
          v [exec_kv]
    k_pe              (T_local, 1, 64)
    k_nope            (T_local, 1, 512)
          |
          v [all_gather KV] + [写入 cache]
    fused_kv_all      (T_total, 704)
          |
          v [q_proj + W_UK_T bmm]
    ql_nope           (T_local, 128, 512)
    q_pe              (T_local, 128, 64)
          |
          v [indexer + sparse_flash_attention]
    attn_output       (T_local, 128, 512)
          |
          v [_v_up_proj: bmm × W_UV]
    attn_output_v     (T_local, 16384)
          |            (128 heads × 128 dim)
          |
          v [view + permute + reshape]
    send              (tp, T_local, 4096)
          |            (拆分为 tp 个 head-chunk)
          |
          v [all_to_all_single]
    attn_output       (T_total, 4096)
          |            (T_total tokens × N_per_rank heads)
          |
          v [o_proj: RowParallelLinear]
    output            (T_total, 7168)
                     [all-reduce inside RowParallelLinear]
```

**代码位置索引**（按 forward 流程顺序）：

| 步骤 | 说明 | 文件 | 关键行号 |
|------|------|------|---------|
| 模式判定 | `full_gather_o_proj_enabled = False`（DecodeOnly/SpecDecoding） | `attention/sfa_v1.py` | `L1098-L1101` |
| DSA-CP 元数据 | 取 `slot_mapping_cp`、`actual_seq_lengths_query/key` | `attention/sfa_v1.py` | `L1077-L1085` |
| KV All-Gather | `all_gather_async`，`async_op` 由 `full_gather_o_proj_enabled` 控制 | `attention/sfa_v1.py` | `L1143-L1157` |
| Q Projection | `_q_proj_and_k_up_proj` | `attention/sfa_v1.py` | `L816-L828` |
| Sparse Flash Attention | `_execute_sparse_flash_attention_process` | `attention/sfa_v1.py` | `L1030-L1053` |
| V Up Projection | `_v_up_proj`，输出 `(T_local, 16384)` | `attention/sfa_v1.py` | `L830-L848` |
| O 路径分派 | `if self.enable_dsa_cp_with_o_proj_tp:` → 进 all-to-all 路径 | `attention/sfa_v1.py` | `L1264-L1277` |
| All-to-All 核心 | `view/permute/reshape` → `all_to_all_single` | `attention/sfa_v1.py` | `L757-L764` |
| 权重预分配 | `_init_o_proj_tp_full_params`：预分配 `o_proj_full_pool`、保存 TP 量化参数 | `attention/sfa_v1.py` | `L690-L720` |
| Prefill 权重 gather（对比） | all-gather `o_proj_tp_weight` → `o_proj_full_pool` | `attention/sfa_v1.py` | `L1195-L1201` |

**All-to-All 的设计动机**：

> Prefill 阶段（4.2）是**把 TP 权重 all-gather 成全量**（`o_proj_full_pool`），让每卡用全量权重 × 本地 token 子集直接算出完整结果——本质是 CP 方案：token 分片 + 全量权重。
>
> Decode 阶段 token 数极少（T_local 很小），切换权重的开销不可忽视。所以换成**对 activation 做 all-to-all**：把 `(T_local, 128, 128)` 通过 all-to-all 重排为 `(T_total, 32, 128)`，即 **CP → TP 的等价变换**。接下来走标准 TP RowParallelLinear，all-reduce 后每卡得到全部 token 的 hidden states（下游 Attention 需要全量 token 的 KV）。
>
> 一句话总结：**all-to-all 用 activation 维度的通信，换来了 O 矩阵不用切权重。**

---

## 5. 模式对比总结

| 维度 | PD 分离 P 节点 | PD 分离 D 节点 | PD 混部 Prefill | PD 混部 Decode |
|------|---------------|---------------|----------------|---------------|
| DSA-CP | 启用 | 不启用 | 启用 | 启用 |
| Token 切分 | 是（T_local） | 否（T_total） | 是（T_local） | 是（T_local） |
| local_num_heads | 128 | 32 (tp=4) | 128 | 128 |
| KV 通信 | all-gather | 无 | all-gather | all-gather |
| V_up_proj 输出 | (T_local, 16384) | (T, 4096) | (T_local, 16384) | (T_local, 16384) |
| O 矩阵策略 | Layer Sharding（广播） | 标准 TP（all-reduce） | 全量权重切换 | All-to-All + 标准 TP |
| O 输入 | (T_local, 16384) | (T, 4096) | (T_local, 16384) | (T_total, 4096) |
| O 输出 | (T_local, 7168) | (T, 7168) | (T_local, 7168) | (T_total, 7168) |
| 通信原语 | broadcast | all-reduce | all-gather (weight) | all-to-all (activation) + all-reduce |

---

## 6. 设计思路总结

### 6.1 核心矛盾：DSA-CP 引入的 Shape 不匹配

DSA-CP（Context Parallelism）的设计目标是在 prefill 阶段加速 attention：每卡做 KV all-gather 拿到全量 KV，本卡只需处理自己那份 token 子集（T_local = T_total / tp），但在 attention 中拥有全量 128 个 head。这意味着 V Up Projection 输出 `(T_local, N_total × V) = (T_local, 16384)`。

问题出在 O Projection。o_proj 是按标准 TP 设计的 RowPa rallelLinear，它期望的输入是 `(T_total, N_per_rank × V) = (T_total, 4096)`——即**全量 token × 分片 head**。

于是产生了一个形状冲突：

```
DSA-CP 输出: (T_local, 16384)   ← 少 token × 全 head (CP 态)
o_proj 期望: (T_total, 4096)    ← 全 token × 分 head (TP 态)
```

所有四种场景的不同 O 矩阵策略，本质上都是在**解决这个 CP 态到 TP 态的转换**。不同场景选择不同方案，取决于一个核心变量：**token 批量大小**。

### 6.2 Prefill vs Decode 的本质差异

同一模型的两个阶段，计算特性截然不同：

| 维度 | Prefill | Decode |
|------|---------|--------|
| 每卡 token 数 (T_local) | 数百~数千 | 1~数十 |
| 计算特征 | Compute-bound | Memory-bound |
| KV cache 写入 | 大量写入 | 少量追加 |
| 通信能否被计算掩盖 | 可以（计算量大） | 很难（计算量太小） |
| 单位 token 的通信成本 | 摊销到多 token，便宜 | 每 token 分摊极大，昂贵 |

**这条分界线是全部设计选择的总纲**。下面逐个场景展开。

### 6.3 逐场景策略原理分析

#### 6.3.1 D 节点（PD 分离）：不开 DSA-CP —— 从源头消解问题

D 节点是 decode-only，T_local 只有 1~数十。此时 **DSA-CP 本身就不划算**：

- DSA-CP 需要 FlashComm1 → 每层做 KV all-gather。尽管 KV all-gather 的数据量随 T 线性缩放（T_local × 704 elms），但 T 太小时通信 latency 无法被掩盖
- 即使拿到全量 head，对 decode 的 attention 质量提升也微乎其微（输入 token 就那几个）
- 收益远小于代价

所以 D 节点**不设 `VLLM_ASCEND_ENABLE_FLASHCOMM1`**，`enable_dsa_cp()` 返回 False，`local_num_heads = N_per_rank = 32`。V Up Projection 输出 `(T, 4096)`，天然匹配 o_proj 的 TP 输入。**shape 冲突从源头消失了**——这是成本最低的解法。

> **适用条件**：D 节点独立部署，不需要处理大 batch prefill。

#### 6.3.2 P 节点（PD 分离）：Layer Sharding —— 权重分散 + 计算掩盖

P 节点是 prefill-only，T_local 数百~数千，计算量大。此时 **DSA-CP 是必然选择**（prefill 需要全量 KV + 全量 head 做 attention），shape 冲突必须解决。

Layer Sharding 的核心思路：

```
每层 o_proj 权重只存在 rank[layer_idx % tp] 上
  → 节省 3/4 的 o_proj 显存（tp=4 时）
  → 计算到该层时，提前 1 层发起 async broadcast（prefetch_step=1）
  → broadcast 完成时，刚好该层 V_up_proj 也算完
  → 本卡用全量权重 × 本卡 token → 直接出结果
  → 不需要 all-reduce（每卡 token 不同，结果各自有效）
```

**为什么不用 all-to-all？**

P 节点 T_local 很大（比如 1024），all-to-all 的 activation 数据量 = T_local × tp × N_per_rank × V = 1024 × 4 × 4096 ≈ 16.8M elms ≈ 33MB。虽然不算天文数字，但 **all-to-all 必须等 V_up_proj 算完才能开始**，无法 overlap。而 broadcast 可以提前发起（prefetch），被 attention + V_up_proj 的计算遮盖。

**为什么不全量权重 all-gather？**

全量 all-gather 数据量 = 117M elms ≈ 234MB（fp16），是 broadcast 的 4 倍。更重要的是它占用额外显存（`o_proj_full_pool`），对 P 节点这种纯 prefill 实例来说，显存是稀缺资源。

> **适用条件**：P 节点独立部署，内存预算紧，计算量大可以掩盖通信。

#### 6.3.3 PD 混部 Prefill：全量权重 All-Gather —— 双流并行的 CP 方案

PD 混部场景下，同一个进程既要处理 prefill 又要处理 decode。**DSA-CP 对 prefill 是必须的，而 DSA-CP 又不能在运行时动态开关**——attention backend 在 init 时就已经选定了。因此 decode 也必须忍受 DSA-CP 带来的 shape 冲突。

对 Prefill 阶段来说，计算量大，通信可以掩盖。策略是：

```
同时发起两条 all-gather（async）：
  ① KV all-gather:   T_local × 704 elms（小）
  ② o_proj weight all-gather: 117M elms（大）
     ↓ 两者并行传输，被 attention + V_up_proj 的计算遮盖
  ③ wait o_proj_full_handle → 权重已就绪
  ④ set o_proj.weight → 全量权重 → matmul → set 回 TP 权重
```

**这里的一个关键设计细节**：`async_op`（`sfa_v1.py:L1143`）在 PD 混部 prefill 时为 True，允许 KV all-gather 和 o_proj weight all-gather 并行进行。这是 `full_gather_o_proj_enabled=True` 的额外收益。

**为什么不用 Layer Sharding？**

Layer Sharding 要求每卡只存部分层权重，这意味着模型加载时需要跨卡通信。PD 混部 decode 阶段要走标准 TP（见下节），需要每卡都有 o_proj TP 权重。Layer Sharding 和标准 TP 的权重需求冲突——混部场景下每卡必须保留完整的 TP 权重。

> **适用条件**：混部进程必须在 prefill 和 decode 之间来回切换，权重切换（`o_proj_full_pool ↔ o_proj_tp_weight`）是运行时代价，但被 prefill 的计算掩盖了。

#### 6.3.4 PD 混部 Decode：All-to-All —— 通信量最低的 CP→TP 转换

这是整个方案里最精妙的一步。Decode 阶段 T_local 极小（可能只有 1~8），直接套用 prefill 的权重 all-gather 策略会灾难性的：

```
权重 all-gather: 117M elms ≈ 234MB（无论 T 多大，权重是固定的！）
activation all-to-all: T_local × tp × N_per_rank × V ≈ 1 × 4 × 4096 × 2 ≈ 33K elms ≈ 66KB
```

**234MB vs 66KB，差了 3500 倍**。这就是为什么 decode 必须走 activation 通信。

All-to-All 做的是 **CP→TP 重分布**：

```
all-to-all 前 (CP 态):
  每个 rank: (T_local, N_total × V) = (T_local, 16384)
  语义: 本地 token × 全量 head
  → view/permute/reshape → (tp × T_local, N_per_rank × V) = (4×T_local, 4096)
  逻辑: 拆成 tp 个 chunk，chunk[j] 对应 head 组 j 的输出

all-to-all 后 (TP 态):
  每个 rank: (tp × T_local, N_per_rank × V) = (T_total, 4096)
  语义: 从各 rank 收到了 token chunk[本卡rank] → 现在拥有全量 token × 本卡 head
```

本质上是在 token 维度和 head 维度之间做了一个 transposition：token 维度从各卡 gather，head 维度 scatter 到各卡。输入 `(T_local, 16384)` → 输出 `(T_total, 4096)`，恰好匹配 o_proj 的 TP 输入。

然后再走标准 RowParallelLinear → all-reduce → `(T_total, 7168)`。最后的 all-reduce 不可避免（因为每卡需要全量 token 的 hidden states 给下一层 Attention），但 T_total 在 decode 时很小，成本可控。

**为什么 PD 混部 decode 不关掉 DSA-CP？**

DSA-CP 是在模型初始化时决定的（`enable_dsa_cp()` 的结果存入 `self.enable_dsa_cp`），attention backend 的选择、`local_num_heads` 的设置、KV cache 的布局都在 init 时固化。运行时无法在 prefill 和 decode 之间动态切换。因此 decode 必须面对 DSA-CP 引入的 shape 冲突，而 all-to-all 是解决这个冲突的最优方案。

### 6.4 通信量量化对比

以 DSV32 @ tp=4 为例：

| 策略 | 应用场景 | 通信量（单层） | 能否被计算掩盖 | 额外显存 |
|------|---------|---------------|---------------|---------|
| DSA-CP 不开 | D 节点 | 0（无 O 矩阵通信） | — | 0 |
| Layer Sharding (broadcast) | P 节点 | ~29M elms ≈ 58MB | 可以（prefetch） | 省 3/4 o_proj |
| 全量权切 All-Gather | 混部 Prefill | ~117M elms ≈ 234MB | 可以（双流并行） | 234MB (full_pool) |
| Activation All-to-All | 混部 Decode | ~T_local × 32K elms ≈ 66KB (T=1) | 不需要（量极小） | 0 |

**通信量对比（混部 Decode，T_local=1）：**

```
All-Gather 权重: 234 MB
All-to-All 激活:  0.066 MB
                  -------
比值:               3545×
```

这正是 all-to-all 策略存在的根本理由。

### 6.5 设计哲学总结

四条策略的选择可以用一个决策图概括：

```
                    ┌─ Prefill-only (T大) ──→ Layer Sharding
                    │   (通信可掩盖 + 显存节省优先)
PD 分离 ────────────┤
                    │
                    └─ Decode-only (T小) ──→ 不开 DSA-CP
                        (源头消解, 成本最低)

                    ┌─ Prefill (T大) ──→ 全量权重 All-Gather
                    │   (通信可掩盖 + 显存允许)
PD 混部 (DSA-CP=ON) ┤
                    │
                    └─ Decode (T小) ──→ Activation All-to-All
                        (通信量 1/3500, 不可掩盖就做最少)
```

其中最关键的一刀是 **T 的大小**（compute-bound vs memory-bound），第二刀是**是否必须开 DSA-CP**（独立部署 vs 混部）。

最终的四条路径覆盖了所有场景，没有一条是凑合的：

- **D 节点不开 CP**：最省事，从源头消除问题
- **P 节点 Layer Sharding**：最省显存，通信被计算完美掩盖
- **混部 Prefill 权重切换**：唯一可行的 CP 方案，双流并行最大化 overlap
- **混部 Decode all-to-all**：通信量最小化，把 CP 态优雅地转成 TP 态

---

## 7. 混合并行：SFACP 超长序列场景

前面四章分析的都是 `AscendSFAImpl`（基础 SFA），它的 CP 和 TP 是**互斥的二选一**：要么开 DSA-CP（CP 态）、要么不开（TP 态）。但 DSV32 实际上还支持**第三种模式**：CP 和 TP 同时存在、各司其职。

### 7.1 选中条件：`enable_cp()`

```python
# attention/sfa_v1.py:97-102
@staticmethod
def get_impl_cls():
    if enable_cp():
        return AscendSFACPImpl    # ← 超长序列 CP 实现
    return AscendSFAImpl          # ← 基础 SFA（前面四章的分析对象）
```

`enable_cp()` 的定义（`attention/utils.py:59-61`）：

```python
def enable_cp():
    prefill_config = get_current_vllm_config().parallel_config
    return (prefill_config.prefill_context_parallel_size > 1
            or prefill_config.decode_context_parallel_size > 1)
```

注意这是**独立于 `enable_dsa_cp()` 的另一套开关**：

| 条件 | 控制什么 | 触发方式 |
|------|---------|---------|
| `enable_dsa_cp()` | DSA-CP（KV all-gather + 全量 head） | FlashComm1 环境变量 / Dynamic EPLB |
| `enable_cp()` | 超长序列 CP（KV 分块存储 + 跨 CP 组 gather） | `prefill/decode_context_parallel_size` 配置 |

两个开关正交，产生四种组合：

```
enable_cp()=False          enable_cp()=True
─────────────────────────────────────────────────
enable_dsa_cp()=False  │  标准 TP       │  长序列 CP + TP   │  (TP 分头 + CP 分 KV)
enable_dsa_cp()=True   │  DSA-CP        │  DSA-CP + 长序列  │  (全量头 + CP 分 KV)
                       │  (前四章分析)   │  CP                │
```

### 7.2 层次化并行架构

`AscendSFACPImpl`（`attention/context_parallel/sfa_cp.py:164`）继承自 `AscendSFAImpl`，新增：

```python
# sfa_cp.py:199-205
self.pcp_size = get_pcp_group().world_size  # Prefill CP 子组大小
self.dcp_size = get_dcp_group().world_size  # Decode CP 子组大小
```

并行层次示意（world=8, pcp_size=2, tp_size=4）：

```
World = 8 GPUs
│
├── CP Group 0 (ranks 0-1): pcp_size=2
│   ├── rank 0: tp_rank=0, KV blocks[0::2]
│   └── rank 1: tp_rank=1, KV blocks[1::2]
│
├── CP Group 1 (ranks 2-3): pcp_size=2
│   ├── rank 2: tp_rank=2, KV blocks[0::2]
│   └── rank 3: tp_rank=3, KV blocks[1::2]
│
├── CP Group 2 (ranks 4-5): pcp_size=2
│   ...
└── CP Group 3 (ranks 6-7): pcp_size=2
    ...
```

- **CP 层**（token 维度）：KV 分块存储，组内 `all_gather` 拿全量 KV
- **TP 层**（head 维度）：o_proj 权重按 TP 切分，`RowParallelLinear` + all-reduce

### 7.3 与基础 SFA 的关键差异

`AscendSFACPImpl` **只覆盖了 6 个方法**，其余全部继承自 `AscendSFAImpl`：

| 覆盖方法 | 差异 | 行号 |
|---------|------|------|
| `exec_kv()` | pcp_size>1 时 KV 不直接写 cache，先 all_gather 再 reshape_and_cache | `sfa_cp.py:514-540` |
| `_execute_sparse_flash_attention_process()` | decode/prefill 分开处理，prefill 做 Q head/tail split | `sfa_cp.py:207-299` |
| `indexer_select_post_process()` | 同上，Indexer 也分 decode/prefill | `sfa_cp.py:377-483` |
| `gather_kv_cross_cp()` | decode 路径：跨 CP 组 gather KV（`get_dcp_group().all_gather`） | `sfa_cp.py:345-355` |
| `gather_kv_cross_cp_compact()` | prefill 路径：按 valid_block_ids 紧凑 gather KV | `sfa_cp.py:357-364` |
| `_get_full_kv()` | pcp_size>1 时跨 PCP 组 gather K | `sfa_cp.py:542-549` |

**关键设计**：这些覆盖**全部集中在 attention 计算阶段**。`forward()` 本身没有被覆盖，O Projection 路径（`_v_up_proj`、`_handle_o_proj_weight_switch_and_forward`、`self.o_proj`）完全继承自基类。这意味着 **第六章分析的四条 O 矩阵策略对 SFACP 同样适用**。

### 7.4 Prefill Q Head/Tail Split

这是 SFACP 最有特色的优化。当 `pcp_size > 1` 时，prefill 的 Q 被拆成两半（`sfa_cp.py:264-298`）：

```
每卡的 Q token 子集: T_local 个 token
  ├── q_head (head_idx): attend KV 前半段 → q_head_output
  └── q_tail (tail_idx): attend KV 后半段 → q_tail_output
                            ↓
  cat + index_select(q_full_idx) → 恢复原始顺序
```

**为什么这么做？**

超长序列下（比如 128K tokens），KV cache 按 CP 分块存储。每个 CP 组成员拿到的 KV all-gather 结果是**全量 KV**。直接让每个 token attend 全量 KV 太贵。Q head/tail split 让每半 Q 只 attend 自己负责的那半 KV，将 attention 计算量减半，同时保持精度。`q_full_idx` 记录了原始顺序，最后恢复排列即可。

### 7.5 通信路径差异

| 维度 | 基础 SFA (`AscendSFAImpl`) | SFACP (`AscendSFACPImpl`) |
|------|--------------------------|--------------------------|
| KV all-gather 的 group | `get_tp_group()`（全部 TP rank） | `get_pcp_group()` / `get_dcp_group()`（CP 子组） |
| CP group size | = tp_size | = pcp_size / dcp_size（≤ tp_size） |
| KV cache 存储 | 每卡全量（DSA-CP）或分片（TP） | CP 组内分块（每卡 1/pcp_size） |
| Prefill attention | 单次 SFA | Q head/tail split → 两次 SFA |
| O 矩阵行为 | 四种策略 | **相同**（继承 forward） |

> 通信量的差异取决于实际部署配置。例如 world=8, tp=4, pcp=2：基础 SFA 的 KV all-gather 跨 4 卡，SFACP 的 KV all-gather 只跨 PCP 组的 2 卡，通信量减半。

### 7.6 典型部署拓扑

| 场景 | 配置 | 使用类 | O 矩阵策略 |
|------|------|--------|----------|
| 标准 TP（短序列） | tp=4, cp=1 | `AscendSFAImpl` | 标准 TP |
| PD 分离 P（短序列） | tp=4, cp=1, DSA-CP=ON | `AscendSFAImpl` | Layer Sharding |
| PD 混部（短序列） | tp=4, cp=1, DSA-CP=ON | `AscendSFAImpl` | 权重切换 / all-to-all |
| **超长序列 CP+TP** | tp=4, pcp=2, dcp=2 | `AscendSFACPImpl` | 继承自 base，按 `enable_dsa_cp` 决定 |
| **超长序列 DSA-CP** | tp=4, pcp=2, DSA-CP=ON | `AscendSFACPImpl` | 同时有全量 head + CP 分块 KV |

### 7.7 设计小结

SFACP 在基础 SFA 之上加了一层**正交的并行维度**：

```
Token 维度: CP (pcp_size/dcp_size) → KV 分块 + 组内 all_gather
Head 维度:  TP (tp_size)           → 权重切分 + all-reduce

两维正交 → 总并行度 = pcp_size × tp_size
```

O 矩阵不需要特殊处理，因为 SFACP 完全继承了 `forward()` 中的 O 路径。CP 层的 KV gather 发生在 attention 内部（`gather_kv_cross_cp`），输出的 `attn_output` shape 仍然是 `(T_local, N, L)`，其中 N = `local_num_heads`（由 TP 和 DSA-CP 共同决定）。后续 V_up_proj 和 o_proj 的 shape 变化与基础 SFA 完全一致。

---

## 附录：关键文件索引

| 文件 | 说明 |
|------|------|
| `vllm_ascend/attention/sfa_v1.py` | SFA v1 核心实现 |
| `vllm_ascend/ops/linear_op.py` | ShardedCP Column/Row Parallel Op |
| `vllm_ascend/ops/layer_shard_linear.py` | Layer Sharding 机制 |
| `vllm_ascend/utils.py:1268-1307` | enable_dsa_cp/layer_shard/o_proj_tp 判定 |
| `vllm_ascend/distributed/utils.py:39-48` | all_gather_async 实现 |
| `vllm/model_executor/models/deepseek_v2.py` | DeepSeek V3 模型定义 |
| `vllm/model_executor/layers/mla.py` | MultiHeadLatentAttentionWrapper |
| `vllm/attention/layer.py` | MLAAttention 层定义 |
