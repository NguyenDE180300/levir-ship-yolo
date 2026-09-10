# LEVIR-Ship ASRM YOLOv8 variants

## RGB+Saturation + selected-ASRM context refine

`yolov8_baseline_gap_factorized_k15_rgb_saturation_asrm_k4k12k16_context_refine.yaml`
(case `baseline_gap_factorized_k15_rgb_saturation_asrm_k4k12k16_context_refine`) combines
only two existing pieces: the RGB+Saturation, FPN-only, GAP, P2-only Detect topology
from `baseline_gap_factorized_k15_rgb_saturation`, and the image-guided P2/P3
context-refinement path from `input_guided_context_refine_p2p3`.

The ASRM branch sees the untouched raw RGB image, converts it to grayscale, and uses
only the per-directional-family train-split winners **K4, K12, K16**. It produces an
attention prior which is checked against P2/P3 detail and P4/P5 context; the raw SRM
residual is never added directly to P2 or P3. Both refinement gains start at zero,
therefore the initial detector is exactly the RGB+Saturation baseline. Training keeps
Factorized TAL (`kappa=1.5`), GAP, P2-only detection and `mosaic=0`.

## P2′-guided P3/P4 ablation group

Bốn config sau kế thừa trực tiếp graph của
`yolov8_input_guided_context_refine_p2p3.yaml`: backbone YOLOv8n-P2, raw-image ASRM
branch, P2′/P3′ context-refine và Detect P2/P3/P4/P5 đều được giữ nguyên.

| Case | Module mới giữa context-refine và neck | Vai trò của P2′ |
|---|---|---|
| `p2guided_gate_p3p4` | spatial + channel residual modulation | sinh gate cho P3′/P4 |
| `p2guided_dynamic_conv_p3p4` | mixture của 4 depthwise 3×3 experts | sinh hệ số mixing |
| `p2guided_cross_attention_p3p4` | local-window cross-attention 3×3 | Key/Value cho Query P3′/P4 |
| `p2guided_deform_conv_p3p4` | modulated deformable conv 3×3 | sinh offset và mask |

Mỗi module nhận `[target, P2′]`, giữ nguyên shape target và có residual scale
`gamma=0`, nên graph mới bắt đầu bằng đúng feature của base. P5 và backbone trunk
không đổi; raw P2′/ASRM residual không được inject vào P3/P4. Code chung nằm tại
`ultralytics/nn/modules/p2_guided_ops.py`.

Ba kiến trúc gốc (Version 1/2/3), cùng thay đổi backbone/neck theo hướng SRM (Spatial
Rich Model) để tăng tín hiệu vật thể nhỏ. Version 1 và 3 mỗi cái có **thêm một biến thể
riêng, độc lập** áp ASRM lên cả P2/8 và P3/8 (hai tap độc lập, cùng class module, mỗi
tap chỉ sửa đúng level của nó) — không thay thế bản P2-only gốc mà tồn tại song song
như một case khác trong ma trận (`p2p3_asrm_fusion`, `input_asrm_guided_p2p3`). Version
2 chỉ có bản P2. Tổng cộng **5 file yaml kiến trúc** (chưa tính `baseline_p2` là config
gốc của Ultralytics). Tất cả dùng chung `ultralytics/nn/modules/asrm.py`
(`AdaptiveSRM`, `P2ASRMFusion`, `InputASRMAuxiliaryAttention`,
`InputASRMAttentionGuidedP2`), đã đăng ký vào `ultralytics/nn/modules/__init__.py` và
`ultralytics/nn/tasks.py` (import + 1 nhánh `elif` trong `parse_model` để tự suy ra số
kênh P2, giống cách Ultralytics tự làm cho `Concat`/`CBFuse`/... — không đụng vào vòng
lặp forward lõi).

`AdaptiveSRM` khởi tạo từ 3 kernel SRM tốt nhất đo được trên LEVIR-Ship
(`mmdet_code/eda_srm_overlap.py`, K4/K5/K8 — filter khác biệt bậc một, 2 tap, hướng
chéo/dọc/chéo, tỉ lệ trúng tàu nhỏ 100%). `srm_mode="adaptive"` (mặc định) cho phép các
trọng số này fine-tune tiếp qua backprop; `srm_mode="fixed"` đóng băng đúng giá trị SRM.

## Version 1 — `yolov8_p2_asrm_fusion.yaml` (P2 only)

```
P2 → AdaptiveSRM (depthwise) → project → gated fusion (P2, residual) → P2'
```

`P2' = P2 + gamma * gate * R2`, `gamma` learnable zero-init nên model khởi động giống
hệt baseline P2. `P2'` chỉ thay P2 ở nhánh lateral vào neck (layer 18); nhánh trục chính
tạo P3/P4/P5 vẫn đọc P2 gốc — P3-P5 không đổi.

`P2ASRMFusion(channels, srm_mode="adaptive", gate_type="spatial_channel", gamma_init=0.0,
gamma_channel_wise=True, debug=False)`. `gate_type` ∈ {`spatial`, `channel`,
`spatial_channel`}. `channels` được `parse_model` tự truyền vào từ đồ thị — không khai
trong yaml.

### Version 1 mở rộng — `yolov8_p2p3_asrm_fusion.yaml` (P2 + P3), case `p2p3_asrm_fusion`

```
P2 → AdaptiveSRM (depthwise) → project → gated fusion (P2, residual) → P2'
P3 → AdaptiveSRM (depthwise) → project → gated fusion (P3, residual) → P3'
```

Hai tap **độc lập** của cùng module `P2ASRMFusion` (module không hardcode P2 trong code,
chỉ cần channel count — tái dùng nguyên class cho cả hai level thay vì viết class mới).
Mỗi tap có `gamma`/`gate` riêng, cả hai zero-init nên model khởi động giống hệt baseline.
`P2'` thay P2 ở nhánh lateral P2 (layer 19); `P3'` thay P3 ở nhánh lateral P3 (layer 16).
Nhánh trục chính (backbone) vẫn luôn đọc P2/P3 **gốc** để tạo các level sâu hơn — P4/P5
không đổi. Đây là **case riêng, tách biệt** khỏi `p2_asrm_fusion` — cả hai cùng tồn tại
trong ma trận, không cái nào bị xóa hay thay thế cái kia.

## Version 2 — `yolov8_input_asrm_auxiliary.yaml` (P2 only)

Nhánh auxiliary theo tinh thần DRE-Net: SRM chạy trên **ảnh gốc** (grayscale), không
chạy trên P2. Không fusion ngược vào P2 — detection path giống hệt baseline P2 (đã kiểm
bằng test: neck lateral vẫn `Concat[-1, <P2 gốc>]`, không đụng layer aux).

```
image → grayscale → AdaptiveSRM(3 kernel) → resize về P2 → project
P2 → project ─┐
              cross-attention (window hoặc CBAM) → attention_map (sigmoid)
```

Module cache kết quả trên `module.last_attention_map` (và, ở `attention_type="window"`,
`module.last_attention_feature` — feature trước sigmoid) sau mỗi forward, để training
loop bên ngoài đọc và tự ghép vào auxiliary loss; **repo này không tự tính loss đó**.
`enable_auxiliary_in_eval=False` (mặc định) tắt hẳn nhánh khi `model.eval()`, nên chi phí
và output lúc inference giống hệt baseline.

`InputASRMAuxiliaryAttention(p2_channels, embed_channels=64, srm_mode="adaptive",
attention_type="window"|"cbam", window_size=8, enable_auxiliary_in_eval=False)`.

## Version 3 — `yolov8_input_asrm_guided_p2.yaml` (P2 only)

Attention map (từ ảnh gốc + P2) dùng để enhance P2 trực tiếp, chạy cả train lẫn eval:

- `fusion_mode="attention_only"`: `P2_out = P2 * (1 + gamma * A2)`
- `fusion_mode="residual_injection"` (mặc định): `P2_out = P2 + gamma * A2 * R2_proj`

`gamma` learnable zero-init → model khởi động giống baseline. `P2'` (guided) thay P2 gốc
ở nhánh lateral vào neck; trục chính P3-P5 vẫn dùng P2 gốc, không đổi.

`InputASRMAttentionGuidedP2(p2_channels, embed_channels=64, srm_mode="adaptive",
attention_type="window"|"cbam", fusion_mode="residual_injection", gamma_init=0.0,
window_size=8, debug=False)`.

### Version 3 mở rộng — `yolov8_input_asrm_guided_p2p3.yaml` (P2 + P3), case `input_asrm_guided_p2p3`

Hai tap **độc lập** của cùng module `InputASRMAttentionGuidedP2` (P2 và P3), mỗi tap
lấy ảnh gốc + level tương ứng, attention map dùng để enhance chính level đó trực tiếp,
chạy cả train lẫn eval. `gamma` learnable zero-init riêng cho từng tap → model khởi
động giống baseline. `P2'` thay P2 gốc ở nhánh lateral P2 (layer 20); `P3'` thay P3 gốc
ở nhánh lateral P3 (layer 17). Trục chính backbone luôn đọc P2/P3 **gốc** để tạo P4/P5
— P4/P5 không đổi. Đây là **case riêng, tách biệt** khỏi `input_asrm_guided_p2` — cả
hai cùng tồn tại trong ma trận, không cái nào bị xóa hay thay thế cái kia.

`InputASRMAttentionGuidedP2(level_channels, embed_channels=64, srm_mode="adaptive",
attention_type="window"|"cbam", fusion_mode="residual_injection", gamma_init=0.0,
window_size=8, debug=False)`.

## Cách 2 module ảnh+P2 lấy được ảnh gốc mà không sửa lõi Ultralytics

`ultralytics.nn.tasks.BaseModel._predict_once` đã hỗ trợ sẵn `from` là một list (dùng
cho `Concat`): `x = [x if j == -1 else y[j] for j in m.f]`. Version 2/3 tận dụng đúng cơ
chế này: layer 0 của backbone là `nn.Identity` chỉ để "giữ" input gốc lại trong `y[0]`;
module ASRM khai `f: [p2_index, 0]` để nhận `[p2, image]`. Không cần viết wrapper Model
riêng, không sửa vòng lặp forward của Ultralytics — chỉ thêm 1 file module mới + đăng ký
tên trong `modules/__init__.py`/`tasks.py`, đúng cách Ultralytics tự mở rộng cho mọi
module đặc biệt khác (`Concat`, `CBFuse`, `TorchVision`...).

Do có `nn.Identity` chèn ở đầu, **toàn bộ chỉ số layer của các file V2/V3 lệch so với
`yolov8-p2.yaml` gốc** — xem comment số layer trong từng file. Layer `Upsample` đầu tiên
của head phải trỏ **tường minh** về SPPF thay vì `-1` ngầm định, vì các tap ASRM được
chèn ngay sau backbone/trước head nên `-1` sẽ trỏ nhầm sang tap cuối cùng.
`p2_asrm_fusion`/`p2p3_asrm_fusion` (V1, không có `nn.Identity`) có SPPF ở layer 9 nên
head trỏ `[9, ...]`; `input_asrm_auxiliary`/`input_asrm_guided_p2`/
`input_asrm_guided_p2p3` (có `nn.Identity` cộng 1-2 tap) có SPPF ở layer 10 nên head trỏ
`[10, ...]`.

## Pretrained-weight transfer (yolov8n.pt)

## Candidate verification + sparse refinement

`yolov8_asrm_candidate_sparse_refine_p2p3.yaml` implements the high-recall candidate
pipeline. Absolute K4/K5/K8 responses are normalized, reduced to 3x3 local maxima and
top-64 candidates, and retained only above the adaptive-response threshold. Candidate
detail is sampled from projected P2/P3. P4/P5 are fused and softly assigned to eight
learned background prototypes; maximum cosine similarity gives background likeness,
and `0.5 * (1 - similarity)` gives novelty. A verifier MLP combines K4/K5/K8 evidence,
mean/max/agreement, both detail vectors and novelty to predict `q_i`.

Each accepted candidate produces a Gaussian truncated at three sigma; max aggregation
forms a genuinely local sparse mask. The output is
`P' = P + gamma * M * phi(P)`, independently for P2 and P3. ASRM residual values never
become detection feature content. Channel-wise `gamma=0` makes both paths exact
identities initially. P4/P5 and Detect strides `4/8/16/32` remain unchanged.

## Context-aware ASRM detail/context refinement

`yolov8_asrm_detail_context_refine_p2p3.yaml` uses two independent refiners. Each
gate combines absolute ASRM evidence, target and peer P2/P3 detail, plus P4/P5 semantic
context. The output is only:

```
refined = original_feature * (1 + gamma * context_aware_gate)
```

No raw or projected ASRM residual is injected. P2 and P3 are refined independently;
P4/P5 remain unchanged. Channel-wise `gamma` is initialized to zero, making both
paths exact identities at initialization. Detect remains P2/P3/P4/P5 with strides
4/8/16/32.

`Levir_ship_training_YOLO/run_experiment.py` transfer-learn từ `yolov8n.pt` thay vì
train from scratch, để so sánh công bằng với baseline yolov8n LEVIR-Ship tham chiếu ở
[`reference implementation`](the reference implementation)
(`train_all_levir_yolov8n_p2_routing.py`), vốn cũng transfer-learn. Ultralytics
`Model.load()` khớp tensor theo **đúng tên layer + shape**
(`ultralytics.nn.tasks.BaseModel.load` → `intersect_dicts`), nên với các case không có
`nn.Identity` (backbone layer 0-9 giữ nguyên index như `yolov8n.pt`) chỉ cần gọi thẳng
`.load('yolov8n.pt')`. Với các case có `nn.Identity` chèn ở layer 0 (dịch toàn bộ
backbone +1 — xem mục trên), phải tự remap tên layer trước khi load, cùng nguyên lý với
hàm `remap_dbss_backbone` trong repo tham chiếu đó (họ cũng phải remap vì DBSS chèn
thêm 1 layer ở đầu backbone). Đã test: cả 6 case đều transfer đúng tensor backbone
(kiểm bằng so sánh trực tiếp giá trị weight sau khi load với checkpoint gốc).

| Case | Backbone index so với `yolov8n.pt` | Cách transfer |
|---|---:|---|
| `baseline_p2` | giữ nguyên `0..9` | `model.load('yolov8n.pt')` |
| `p2_asrm_fusion` | giữ nguyên `0..9` | `model.load('yolov8n.pt')` |
| `p2p3_asrm_fusion` | giữ nguyên `0..9` | `model.load('yolov8n.pt')` |
| `input_asrm_auxiliary` | dịch thành `1..10` do `Identity@0` | remap `model.i.* → model.(i+1).*` cho backbone |
| `input_asrm_guided_p2` | dịch thành `1..10` do `Identity@0` | remap `model.i.* → model.(i+1).*` cho backbone |
| `input_asrm_guided_p2p3` | dịch thành `1..10` do `Identity@0` | remap `model.i.* → model.(i+1).*` cho backbone |
| `asrm_detail_context_refine_p2p3` | dịch thành `1..10` do `Identity@0` | remap `model.i.* → model.(i+1).*` cho backbone |
| `asrm_candidate_sparse_refine_p2p3` | dịch thành `1..10` do `Identity@0` | remap `model.i.* → model.(i+1).*` cho backbone |

Remap chỉ nhận tensor vừa có key đích vừa khớp shape và sẽ fail-fast nếu số tensor
transfer bằng 0. Detection head/P2/ASRM không có tensor tương ứng trong checkpoint
stock thì giữ initialization của config hiện tại; đây là transfer learning, không phải
resume nguyên một checkpoint đã train trên LEVIR-Ship.

## Phạm vi so sánh với `reference implementation`

Runner đóng gói tại `../Levir_ship_training_YOLO/run_experiment.py` dùng cùng
`yolov8n.pt`, `imgsz=512`, `batch=8` và các seed 42/43/44 như baseline tham chiếu
[`reference implementation`](the reference implementation). Khác biệt chủ đích là
module ASRM/P2. Tuy nhiên pipeline này giữ split theo **scene** (2728/584/584), còn
script tham chiếu shuffle/cắt ngẫu nhiên theo crop (2320/788/788). Vì vậy:

- dùng `baseline_p2` và các ASRM case còn lại trong cùng matrix này để kết luận delta
  kiến trúc;
- không coi chênh lệch mAP tuyệt đối với bảng của `reference2408` là ablation trực tiếp;
- mọi case trong matrix phải dùng cùng split, pretrained checkpoint, image size, batch,
  epoch, patience và seed.

## Trạng thái đã kiểm chứng

Đã build cả 5 file kiến trúc (`baseline_p2` dùng thẳng config gốc Ultralytics) qua
`ultralytics.nn.tasks.DetectionModel` (không phải chỉ đọc yaml) ở mọi scale
`n/s/m/l/x`, forward+backward trên tensor giả, so khớp shape 4 mức đầu ra P2/P3/P4/P5,
và xác nhận `gamma_init=0` → output giống hệt input ở bước khởi tạo cho tất cả. Đã test
thêm: pretrained-weight transfer đúng tensor cho cả 6 case (mục trên), và train/val
thật (CPU, ảnh thật, 1 epoch) qua `run_experiment.py` không lỗi.
**Chưa train full-scale trên GPU** — đây mới là bước chuẩn bị pipeline, chưa phải kết
quả benchmark. Repo `YOLO_custom_org` vẫn là vanilla Ultralytics `v8.4.67` cộng đúng
các thay đổi liệt kê trong `git diff` (xem lệnh bên dưới), chưa cài `pip install -e .`;
khi train thật, chạy Python từ trong `YOLO_custom_org/` (hoặc set `PYTHONPATH`) để chắc
chắn dùng đúng source local này, không phải bản `ultralytics` đã `pip install` sẵn
trong môi trường (nếu có, ví dụ conda env `tf_linux` đang có sẵn bản pip 8.3.161
khác).

```bash
cd YOLO_custom_org
git diff --stat  # xem đúng các file đã đổi
```

## Chạy chung với dataset khác (Varroa) trong 1 notebook

`../ASRM_matrix_both_datasets.ipynb` (repo root) orchestrate cả `Levir_ship_training_YOLO`
và `Varroa_asrm_training` trong 1 lần chạy: tải code+data của cả hai từ đúng HF repo
riêng, prepare dataset bằng script riêng của từng bên, train tuần tự 2 case
(`p2p3_asrm_fusion`, `input_asrm_guided_p2p3` — 2 biến thể P2+P3) × seed cho từng
dataset (không đổi hyperparameter/split đã verify riêng của mỗi bên), rồi
`combine_results.py` gộp 2 bảng aggregate (gắn cột `dataset`) thành **một** DataFrame.
Không gộp trị số mAP giữa 2 dataset — chỉ đặt cạnh nhau để so kiến trúc. 4 case còn lại
(`baseline_p2`, `p2_asrm_fusion`, `input_asrm_auxiliary`, `input_asrm_guided_p2`) không
train trong notebook này, chỉ được smoke-test kiến trúc; chạy riêng nếu cần so sánh với
baseline.

## Trạng thái pipeline LEVIR-Ship

- `Levir_ship_training_YOLO/prepare_yolo_data.py` đã tạo layout YOLO bằng symlink và
  giữ đúng scene split dùng chung với phía mmdet.
- `run_experiment.py` đã chạy smoke end-to-end cả sáu case bằng pretrained weights;
  output tách theo `work_root/<case>/seed_<seed>/`.
- `collect_results.py` đã gom nhiều seed thành raw CSV và aggregate mean/std CSV, có
  `--cases` để chọn subset (delta vs `baseline_p2` chỉ tính khi case đó có trong subset).
- Notebook độc lập LEVIR-Ship lặp đủ ba seed 42/43/44 với mặc định `imgsz=512`,
  `batch=8`; notebook hợp nhất 2 dataset hiện chỉ train 2 case P2+P3.
- Version 2 vẫn chưa có auxiliary loss thật (chỉ expose `last_attention_map`), nên
  detection path hiện tương đương baseline; muốn nó tạo delta cần bổ sung loss riêng.
## CARAFE P2 experiment

`yolov8_asrm_context_carafe_p2.yaml` is based on
`yolov8_input_guided_context_refine_p2p3.yaml` (image guidance only at P2). It
changes only the final top-down P3-to-P2 upsample from nearest-neighbor to
CARAFE (`scale=2`, `Cm=64`, `kencoder=3`, `kup=5`).
