"""Point Transformer V3 geometry-to-field operator for DrivAerML.

The geometry encoder follows the detached PointTransformerV3 repository:
voxel coordinates are serialized with Morton/Hilbert orders, attention is
performed in serialized patches, and the hierarchy uses sparse 3D CPE plus
serialized pooling/unpooling.  The final operator head is a small query
decoder that maps the encoded geometry representation to arbitrary surface
and volume coordinates.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import torch_scatter

try:
    from torch_cluster import knn as torch_cluster_knn
except ImportError:  # pragma: no cover - only used in minimal environments
    torch_cluster_knn = None

from .family_common import CondInjection, split_surface_volume_predictions
from .ptv3_serialization import encode

try:
    import flash_attn
except ImportError:  # pragma: no cover - exercised only in minimal environments
    flash_attn = None


def _resolve_spconv_algo(name):
    if name is None:
        return None
    if isinstance(name, spconv.ConvAlgo):
        return name
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "native": spconv.ConvAlgo.Native,
        "mask_implicit_gemm": spconv.ConvAlgo.MaskImplicitGemm,
        "implicit_gemm": spconv.ConvAlgo.MaskImplicitGemm,
        "mask_split_implicit_gemm": spconv.ConvAlgo.MaskSplitImplicitGemm,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported PTv3 spconv algorithm: {name!r}")
    return aliases[normalized]


def _query_local_geometry_features(
    query_pos,
    geometry_coords,
    geometry_features,
    geometry_batch,
    neighbors,
    chunk_size,
):
    """Interpolate final PTv3 point features at arbitrary query locations."""
    batch_size = int(query_pos.shape[0])
    output = []
    for batch_index in range(batch_size):
        geometry_mask = geometry_batch == batch_index
        coords = geometry_coords[geometry_mask].float().contiguous()
        features = geometry_features[geometry_mask]
        queries = query_pos[batch_index].float().contiguous()
        if queries.shape[0] == 0:
            output.append(features.new_empty((0, features.shape[-1])))
            continue
        if coords.shape[0] == 0:
            output.append(features.new_zeros((queries.shape[0], features.shape[-1])))
            continue
        k = min(max(1, int(neighbors)), int(coords.shape[0]))
        query_parts = []
        for start in range(0, int(queries.shape[0]), max(1, int(chunk_size))):
            query_chunk = queries[start : start + int(chunk_size)]
            if torch_cluster_knn is not None:
                edge_index = torch_cluster_knn(coords, query_chunk, k=k)
                # torch_cluster.knn(x, y, k) returns [query_index, x_index],
                # i.e. the query ids are row 0 and geometry-neighbor ids are
                # row 1.  Keep every query's k neighbors contiguous before
                # reshaping; interpreting the rows in reverse silently mixes
                # local features and can also create invalid geometry indices.
                query_index = edge_index[0]
                neighbor_index = edge_index[1]
                center_order = torch.argsort(query_index, stable=True)
                query_index = query_index[center_order]
                neighbor_index = neighbor_index[center_order]
                expected_query_index = torch.arange(query_chunk.shape[0], device=query_chunk.device).repeat_interleave(k)
                if not torch.equal(query_index, expected_query_index):
                    raise RuntimeError("torch_cluster.knn did not return k neighbors for every query in order")
                neighbor_index = neighbor_index.view(query_chunk.shape[0], k)
            else:
                distances = torch.cdist(query_chunk, coords).pow(2)
                neighbor_index = torch.topk(distances, k=k, dim=-1, largest=False).indices
            neighbor_coords = coords[neighbor_index]
            distance2 = (query_chunk.unsqueeze(1) - neighbor_coords).pow(2).sum(dim=-1)
            temperature = distance2.detach().mean(dim=-1, keepdim=True).clamp_min(1.0e-8)
            weights = torch.softmax(-distance2 / temperature, dim=-1).to(dtype=features.dtype)
            query_parts.append((weights.unsqueeze(-1) * features[neighbor_index]).sum(dim=1))
        output.append(torch.cat(query_parts, dim=0))
    return torch.stack(output, dim=0)


class _DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor()


@torch.inference_mode()
def _offset_to_counts(offset):
    return torch.diff(offset, prepend=torch.zeros(1, device=offset.device, dtype=torch.long))


class _Point(dict):
    """Minimal Pointcept-compatible point container."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def serialization(self, order, shuffle_orders):
        if "batch" not in self:
            raise KeyError("PTv3 point data requires a batch vector.")
        grid_coord = self.grid_coord
        max_coord = int(grid_coord.max().item()) if grid_coord.numel() else 0
        depth = max(1, max_coord.bit_length())
        if depth * 3 + int(self.offset.numel()).bit_length() > 63:
            raise ValueError("Point coordinates do not fit in PTv3's 63-bit serialization key.")
        if depth > 16:
            raise ValueError("PTv3 serialization supports at most 16 bits per coordinate.")
        codes = [encode(grid_coord, self.batch, depth, order_name) for order_name in order]
        codes = torch.stack(codes)
        serialized_order = torch.argsort(codes, dim=1)
        serialized_inverse = torch.zeros_like(serialized_order).scatter_(
            1,
            serialized_order,
            torch.arange(codes.shape[1], device=codes.device).repeat(codes.shape[0], 1),
        )
        if shuffle_orders:
            permutation = torch.randperm(codes.shape[0], device=codes.device)
            codes = codes[permutation]
            serialized_order = serialized_order[permutation]
            serialized_inverse = serialized_inverse[permutation]
        self.serialized_depth = depth
        self.serialized_code = codes
        self.serialized_order = serialized_order
        self.serialized_inverse = serialized_inverse

    def sparsify(self, pad=96):
        if "grid_coord" not in self:
            self.grid_coord = torch.floor(self.coord / self.grid_size).long()
        if "sparse_shape" not in self:
            self.sparse_shape = (self.grid_coord.max(dim=0).values + int(pad)).tolist()
        indices = torch.cat([self.batch[:, None].int(), self.grid_coord.int()], dim=1).contiguous()
        self.sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=indices,
            spatial_shape=self.sparse_shape,
            batch_size=int(self.batch.max().item()) + 1,
        )
        return self


class _PointModule(nn.Module):
    pass


class _PointDropPath(_PointModule):
    """Apply official DropPath to point features while preserving the container."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop = _DropPath(drop_prob)

    def forward(self, point):
        point.feat = self.drop(point.feat)
        if "sparse_conv_feat" in point:
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class _PointSequential(_PointModule):
    def __init__(self, *modules):
        super().__init__()
        self._ordered = OrderedDict()
        for index, module in enumerate(modules):
            self.add_module(str(index), module)

    def add(self, module, name=None):
        self.add_module(str(len(self._modules)) if name is None else name, module)

    def forward(self, value):
        for module in self._modules.values():
            if isinstance(module, _PointModule):
                value = module(value)
            elif spconv.modules.is_spconv_module(module):
                sparse = value.sparse_conv_feat
                # Sparse FP16 convolution is prone to overflow for the large
                # dynamic range produced by changing CFD point clouds.  Keep
                # the PTv3 sparse kernels and weights in FP32; attention and
                # query decoding remain eligible for AMP.
                if sparse.features.is_cuda:
                    with torch.autocast(device_type="cuda", enabled=False):
                        sparse = sparse.replace_feature(sparse.features.float())
                        sparse = module(sparse)
                else:
                    sparse = module(sparse)
                value.sparse_conv_feat = sparse
                value.feat = value.sparse_conv_feat.features
            elif isinstance(value, _Point):
                value.feat = module(value.feat)
                if "sparse_conv_feat" in value:
                    value.sparse_conv_feat = value.sparse_conv_feat.replace_feature(value.feat)
            else:
                value = module(value)
        return value


class _MLP(nn.Module):
    def __init__(self, channels, hidden_channels, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, channels)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _SerializedAttention(_PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        order_index,
        attn_drop=0.0,
        proj_drop=0.0,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
    ):
        super().__init__()
        if int(channels) % int(num_heads) != 0:
            raise ValueError("PTv3 channels must be divisible by num_heads.")
        if enable_flash and flash_attn is None:
            raise ImportError("enable_flash=True requires flash-attn to be installed.")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.patch_size_max = int(patch_size)
        self.order_index = int(order_index)
        self.enable_flash = bool(enable_flash)
        self.upcast_attention = bool(upcast_attention)
        self.upcast_softmax = bool(upcast_softmax)
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.channels, 3 * self.channels, bias=True)
        self.proj = nn.Linear(self.channels, self.channels)
        self.proj_drop = nn.Dropout(float(proj_drop))
        self.attn_drop = float(attn_drop)
        self.patch_size = self.patch_size_max

    def _padding(self, point):
        key = f"ptv3_padding_{self.order_index}_{self.patch_size}"
        if key in point:
            return point[key]
        counts = _offset_to_counts(point.offset)
        patch_size = int(self.patch_size)
        padded_counts = torch.div(counts + patch_size - 1, patch_size, rounding_mode="trunc") * patch_size
        padded_counts = torch.where(counts > patch_size, padded_counts, counts)
        offset = F.pad(point.offset, (1, 0))
        offset_padded = F.pad(torch.cumsum(padded_counts, dim=0), (1, 0))
        pad = torch.arange(int(offset_padded[-1]), device=point.feat.device)
        unpad = torch.arange(int(offset[-1]), device=point.feat.device)
        cu_seqlens = []
        for index, count in enumerate(counts.tolist()):
            start = int(offset[index])
            stop = int(offset[index + 1])
            padded_start = int(offset_padded[index])
            padded_stop = int(offset_padded[index + 1])
            if int(count) != int(padded_counts[index]):
                tail = int(count) % patch_size
                pad[padded_stop - patch_size + tail : padded_stop] = pad[
                    padded_stop - 2 * patch_size + tail : padded_stop - patch_size
                ]
            unpad[start:stop] += padded_start - start
            pad[padded_start:padded_stop] -= padded_start - start
            cu_seqlens.append(torch.arange(padded_start, padded_stop, patch_size, device=point.feat.device, dtype=torch.int32))
        cu_seqlens = F.pad(torch.cat(cu_seqlens), (0, 1), value=int(offset_padded[-1]))
        point[key] = (pad, unpad, cu_seqlens)
        return point[key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = max(1, min(int(_offset_to_counts(point.offset).min().item()), self.patch_size_max))
        pad, unpad, cu_seqlens = self._padding(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        qkv = self.qkv(point.feat)[order]
        if self.enable_flash:
            packed = qkv.to(dtype=torch.float16).reshape(-1, 3, self.num_heads, self.head_dim)
            features = flash_attn.flash_attn_varlen_qkvpacked_func(
                packed,
                cu_seqlens,
                max_seqlen=int(self.patch_size),
                dropout_p=self.attn_drop if self.training else 0.0,
                softmax_scale=self.scale,
            ).reshape(-1, self.channels).to(dtype=qkv.dtype)
        else:
            k = int(self.patch_size)
            q, key, value = qkv.reshape(-1, k, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
            if self.upcast_attention:
                q, key = q.float(), key.float()
            attention = (q * self.scale) @ key.transpose(-2, -1)
            if self.upcast_softmax:
                attention = attention.float()
            attention = attention.softmax(dim=-1)
            attention = F.dropout(attention, p=self.attn_drop, training=self.training)
            features = (attention @ value).transpose(1, 2).reshape(-1, self.channels).to(qkv.dtype)
        point.feat = self.proj_drop(self.proj(features[inverse]))
        return point


class _SerializedBlock(_PointModule):
    def __init__(
        self,
        channels,
        heads,
        patch_size,
        order_index,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flash=True,
        attn_drop=0.0,
        proj_drop=0.0,
        spconv_algo=None,
        cpe_indice_key=None,
    ):
        super().__init__()
        if cpe_indice_key is None:
            raise ValueError("PTv3 CPE blocks require a stage-specific indice key.")
        self.cpe = _PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=str(cpe_indice_key),
                algo=spconv_algo,
            ),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )
        self.norm1 = _PointSequential(nn.LayerNorm(channels))
        self.attn = _SerializedAttention(
            channels,
            heads,
            patch_size,
            order_index,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            enable_flash=enable_flash,
            upcast_attention=False,
            upcast_softmax=False,
        )
        self.norm2 = _PointSequential(nn.LayerNorm(channels))
        self.mlp = _PointSequential(_MLP(channels, int(channels * float(mlp_ratio)), dropout=proj_drop))
        self.drop_path = _PointDropPath(drop_path)

    def forward(self, point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        shortcut = point.feat
        point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class _SerializedPooling(_PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        shuffle_orders=True,
        preserve_density=False,
        bn_eps=1.0e-3,
        bn_momentum=0.01,
    ):
        super().__init__()
        if int(stride) not in {2, 4, 8}:
            raise ValueError("PTv3 serialized pooling supports strides 2, 4, and 8.")
        self.stride = int(stride)
        self.shuffle_orders = bool(shuffle_orders)
        self.preserve_density = bool(preserve_density)
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = _PointSequential(
            nn.BatchNorm1d(out_channels, eps=float(bn_eps), momentum=float(bn_momentum)),
            nn.GELU(),
        )

    def forward(self, point):
        pooling_depth = (int(math.ceil(self.stride)) - 1).bit_length()
        pooling_depth = min(pooling_depth, int(point.serialized_depth))
        code = point.serialized_code >> (pooling_depth * 3)
        _, cluster, counts = torch.unique(code[0], sorted=True, return_inverse=True, return_counts=True)
        _, indices = torch.sort(cluster)
        index_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[index_ptr[:-1]]
        code = code[:, head_indices]
        order = torch.argsort(code, dim=1)
        inverse = torch.zeros_like(order).scatter_(
            1,
            order,
            torch.arange(order.shape[1], device=order.device).repeat(order.shape[0], 1),
        )
        if self.shuffle_orders:
            permutation = torch.randperm(code.shape[0], device=code.device)
            code, order, inverse = code[permutation], order[permutation], inverse[permutation]
        pooled_features = self.proj(point.feat)[indices]
        pooled_feat = torch_scatter.segment_csr(
            pooled_features,
            index_ptr,
            reduce="sum" if self.preserve_density else "max",
        )
        pooled_coord = torch_scatter.segment_csr(
            point.coord[indices],
            index_ptr,
            reduce="sum" if self.preserve_density else "mean",
        )
        pooled_voxel_count = torch_scatter.segment_csr(
            point.voxel_count[indices].float(), index_ptr, reduce="sum"
        )
        pooled = _Point(
            feat=pooled_feat,
            coord=pooled_coord,
            voxel_count=pooled_voxel_count,
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            batch=point.batch[head_indices],
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=int(point.serialized_depth) - pooling_depth,
            offset=torch.cumsum(torch.bincount(point.batch[head_indices]), dim=0).long(),
            pooling_inverse=cluster,
            pooling_parent=point,
            sparse_shape=(point.sparse_shape[0] // (2 ** pooling_depth) + 96,
                          point.sparse_shape[1] // (2 ** pooling_depth) + 96,
                          point.sparse_shape[2] // (2 ** pooling_depth) + 96),
        )
        pooled = pooled.sparsify()
        return self.norm(pooled)


class _SerializedUnpooling(_PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        bn_eps=1.0e-3,
        bn_momentum=0.01,
    ):
        super().__init__()
        self.proj = _PointSequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=float(bn_eps), momentum=float(bn_momentum)),
            nn.GELU(),
        )
        self.proj_skip = _PointSequential(
            nn.Linear(skip_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=float(bn_eps), momentum=float(bn_momentum)),
            nn.GELU(),
        )

    def forward(self, point):
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)
        return parent


class _PTv3Backbone(_PointModule):
    def __init__(
        self,
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_heads=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_heads=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4.0,
        drop_path=0.3,
        shuffle_orders=True,
        preserve_density=False,
        enable_flash=True,
        attn_drop=0.0,
        proj_drop=0.0,
        spconv_algo="native",
        bn_eps=1.0e-3,
        bn_momentum=0.01,
    ):
        super().__init__()
        self.spconv_algo = _resolve_spconv_algo(spconv_algo)
        self.order = tuple(order)
        self.shuffle_orders = bool(shuffle_orders)
        self.num_stages = len(enc_depths)
        if self.num_stages != len(stride) + 1 or self.num_stages != len(enc_channels) or self.num_stages != len(enc_heads):
            raise ValueError("PTv3 encoder lists must have consistent stage lengths.")
        if len(dec_depths) != self.num_stages - 1 or len(dec_channels) != self.num_stages - 1 or len(dec_heads) != self.num_stages - 1:
            raise ValueError("PTv3 decoder lists must have num_stages-1 entries.")
        self.embedding = _PointSequential(
            spconv.SubMConv3d(
                in_channels,
                enc_channels[0],
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="ptv3_stem",
                algo=self.spconv_algo,
            ),
            nn.BatchNorm1d(enc_channels[0], eps=float(bn_eps), momentum=float(bn_momentum)),
            nn.GELU(),
        )
        drop_values = torch.linspace(0.0, float(drop_path), sum(enc_depths)).tolist()
        self.enc = _PointSequential()
        cursor = 0
        for stage in range(self.num_stages):
            sequence = _PointSequential()
            if stage > 0:
                sequence.add(
                    _SerializedPooling(
                        enc_channels[stage - 1],
                        enc_channels[stage],
                        stride=stride[stage - 1],
                        shuffle_orders=shuffle_orders,
                        preserve_density=preserve_density,
                        bn_eps=bn_eps,
                        bn_momentum=bn_momentum,
                    ),
                    name="down",
                )
            for block_index in range(enc_depths[stage]):
                sequence.add(
                    _SerializedBlock(
                        enc_channels[stage], enc_heads[stage], enc_patch_size[stage],
                        order_index=block_index % len(self.order), mlp_ratio=mlp_ratio,
                        drop_path=drop_values[cursor], enable_flash=enable_flash,
                        attn_drop=attn_drop, proj_drop=proj_drop, spconv_algo=self.spconv_algo,
                        cpe_indice_key=f"stage{stage}",
                    ),
                    name=f"block{block_index}",
                )
                cursor += 1
            self.enc.add(sequence, name=f"enc{stage}")

        dec_channel_list = list(dec_channels) + [enc_channels[-1]]
        dec_values = torch.linspace(0.0, float(drop_path), sum(dec_depths)).tolist()
        self.dec = _PointSequential()
        cursor = 0
        for stage in reversed(range(self.num_stages - 1)):
            sequence = _PointSequential()
            sequence.add(
                _SerializedUnpooling(
                    dec_channel_list[stage + 1],
                    enc_channels[stage],
                    dec_channel_list[stage],
                    bn_eps=bn_eps,
                    bn_momentum=bn_momentum,
                ),
                name="up",
            )
            stage_values = list(reversed(dec_values[cursor : cursor + dec_depths[stage]]))
            for block_index, block_drop in enumerate(stage_values):
                sequence.add(
                    _SerializedBlock(
                        dec_channel_list[stage], dec_heads[stage], dec_patch_size[stage],
                        order_index=block_index % len(self.order), mlp_ratio=mlp_ratio,
                        drop_path=block_drop, enable_flash=enable_flash,
                        attn_drop=attn_drop, proj_drop=proj_drop, spconv_algo=self.spconv_algo,
                        cpe_indice_key=f"stage{stage}",
                    ),
                    name=f"block{block_index}",
                )
            cursor += dec_depths[stage]
            self.dec.add(sequence, name=f"dec{stage}")
        self.output_channels = int(dec_channels[0])

    def forward(self, point):
        point = self.embedding(point)
        point = self.enc(point)
        point = self.dec(point)
        return point


class PointTransformerV3(nn.Module):
    """PTv3 encoder with a query decoder for surface and volume fields."""

    expects_geo_log_density = False
    # The shared trainers use this opt-in to make stateful sparse/normalization
    # updates transactional around a failed mixed-precision batch.
    rollback_buffers_on_nonfinite = True

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        in_channels=3,
        grid_size=0.0078125,
        latent_dim=256,
        num_latents=64,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_heads=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_heads=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4.0,
        drop_path=0.3,
        shuffle_orders=True,
        enable_flash=True,
        attn_drop=0.0,
        proj_drop=0.0,
        spconv_algo="native",
        pos_scale_factor=1.0,
        query_chunk_size=65536,
        local_query_neighbors=4,
        local_query_chunk_size=8192,
        use_local_query_features=True,
        dropout=0.0,
        geometry_input_scale=1.0,
        voxel_density_power=0.0,
        preserve_density=False,
        bn_eps=1.0e-3,
        bn_momentum=0.01,
        **_unused,
    ):
        super().__init__()
        if int(spatial_dim) != 3 or int(in_channels) != 3:
            raise ValueError("PointTransformerV3 expects 3D coordinate features.")
        self.surface_channels = int(surface_channels)
        self.volume_channels = int(volume_channels)
        self.parameter_channels = int(parameter_channels)
        self.grid_size = float(grid_size)
        self.pos_scale_factor = float(pos_scale_factor)
        self.geometry_input_scale = float(geometry_input_scale)
        self.voxel_density_power = max(0.0, float(voxel_density_power))
        self.preserve_density = bool(preserve_density)
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.local_query_neighbors = max(1, int(local_query_neighbors))
        self.local_query_chunk_size = max(1, int(local_query_chunk_size))
        self.use_local_query_features = bool(use_local_query_features)
        self.backbone = _PTv3Backbone(
            in_channels=int(in_channels), order=order, stride=stride,
            enc_depths=enc_depths, enc_channels=enc_channels, enc_heads=enc_heads,
            enc_patch_size=enc_patch_size, dec_depths=dec_depths,
            dec_channels=dec_channels, dec_heads=dec_heads,
            dec_patch_size=dec_patch_size, mlp_ratio=mlp_ratio,
            drop_path=drop_path, shuffle_orders=shuffle_orders,
            preserve_density=self.preserve_density,
            enable_flash=enable_flash, attn_drop=attn_drop, proj_drop=proj_drop,
            spconv_algo=spconv_algo,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )
        feature_dim = self.backbone.output_channels
        self.latent_score = nn.Linear(feature_dim, int(num_latents))
        self.latent_value = nn.Linear(feature_dim, int(latent_dim))
        self.local_geometry_proj = nn.Sequential(
            nn.Linear(feature_dim, int(latent_dim)),
            nn.LayerNorm(int(latent_dim)),
            nn.GELU(),
        )
        self.latent_norm = nn.LayerNorm(int(latent_dim))
        self.geometry_cond = CondInjection(int(latent_dim), int(parameter_channels))
        self.query_embed = nn.Sequential(
            nn.Linear(3, int(latent_dim)),
            nn.LayerNorm(int(latent_dim)),
            nn.GELU(),
            nn.Linear(int(latent_dim), int(latent_dim)),
        )
        self.query_type = nn.Parameter(torch.zeros(2, int(latent_dim)))
        nn.init.normal_(self.query_type, std=0.02)
        self.query_q = nn.Linear(int(latent_dim), int(latent_dim), bias=False)
        self.latent_k = nn.Linear(int(latent_dim), int(latent_dim), bias=False)
        self.latent_v = nn.Linear(int(latent_dim), int(latent_dim), bias=False)
        # Positional information is added to the pooled latent tokens before
        # cross-attention.  Normalize again after that addition so the query
        # and key projections cannot grow into near one-hot attention logits.
        self.latent_attention_norm = nn.LayerNorm(int(latent_dim))
        self.query_norm = nn.LayerNorm(int(latent_dim))
        self.query_mlp = nn.Sequential(
            nn.Linear(int(latent_dim), 4 * int(latent_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * int(latent_dim), int(latent_dim)),
        )
        self.query_cond = CondInjection(int(latent_dim), int(parameter_channels))
        self.output_head = nn.Sequential(
            nn.LayerNorm(int(latent_dim)),
            nn.Linear(int(latent_dim), int(latent_dim)),
            nn.GELU(),
            nn.Linear(int(latent_dim), self.surface_channels + self.volume_channels),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _make_point(self, geo):
        batch_size, point_count, _ = geo.shape
        coords = geo.float().reshape(-1, 3)
        grid_coord = torch.floor(coords / self.grid_size).long()
        grid_coord = grid_coord - grid_coord.amin(dim=0, keepdim=True)
        batch = torch.arange(batch_size, device=geo.device, dtype=torch.long).repeat_interleave(point_count)
        # Pointcept feeds the backbone geometric coordinates directly.  Keep
        # the SMART positional scale for query encoding, but do not multiply
        # sparse backbone features by 100 before the first convolution.
        raw_features = (coords * self.geometry_input_scale).to(dtype=geo.dtype)

        # Pointcept grid-samples before constructing the sparse tensor.  This
        # both matches the official pipeline and guarantees unique sparse
        # indices when the sampled CFD cloud contains multiple points in one
        # voxel.  Mean reduction preserves the local coordinate signal.
        voxel_keys = torch.cat([batch[:, None], grid_coord], dim=1)
        _, inverse = torch.unique(voxel_keys, dim=0, sorted=True, return_inverse=True)
        sort_order = torch.argsort(inverse, stable=True)
        counts = torch.bincount(inverse, minlength=int(inverse.max().item()) + 1)
        index_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coords = torch_scatter.segment_csr(coords[sort_order], index_ptr, reduce="mean")
        raw_features = torch_scatter.segment_csr(
            raw_features[sort_order],
            index_ptr,
            reduce="sum" if self.preserve_density else "mean",
        )
        unique_keys = voxel_keys[sort_order[index_ptr[:-1]]]
        grid_coord = unique_keys[:, 1:]
        batch = unique_keys[:, 0].long()
        offset = torch.cumsum(torch.bincount(batch, minlength=batch_size), dim=0).long()
        point = _Point(
            coord=coords,
            grid_coord=grid_coord,
            feat=raw_features,
            voxel_count=counts.to(dtype=torch.float32),
            batch=batch,
            offset=offset,
            grid_size=self.grid_size,
        )
        point.serialization(order=self.backbone.order, shuffle_orders=self.backbone.shuffle_orders)
        point.sparsify()
        return point

    def encode_geometry(self, geo, params=None):
        point = self.backbone(self._make_point(geo))
        # Keep the global readout in FP32.  The outer trainer autocast context
        # would otherwise run these projections and the long-point softmax in
        # FP16, which unnecessarily loses small geometry weights.
        device_type = point.feat.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            point_features = point.feat.float()
            scores = self.latent_score(point_features)
            values = self.latent_value(point_features)
        if self.preserve_density:
            # Do not normalize away the number of points represented by the
            # sparse voxels.  Softplus gates stay positive, while segmented
            # sums retain both local occupancy and global point mass.  The
            # square-root stabilization avoids a quadratic scale increase
            # without turning this back into an average readout.
            segment_offsets = F.pad(point.offset, (1, 0))
            density = point.voxel_count.float().clamp_min(1.0)
            density = density / torch_scatter.segment_csr(
                density, segment_offsets, reduce="mean"
            ).repeat_interleave(_offset_to_counts(point.offset)).clamp_min(1.0)
            gates = F.softplus(scores)
            if self.voxel_density_power > 0.0:
                gates = gates * density.pow(self.voxel_density_power).unsqueeze(-1)
            latent_mass = torch_scatter.segment_csr(gates, segment_offsets, reduce="sum").clamp_min(1.0e-6)
            latent = torch_scatter.segment_csr(
                gates.unsqueeze(-1) * values.unsqueeze(1),
                segment_offsets,
                reduce="sum",
            ) / latent_mass.sqrt().unsqueeze(-1)
            latent_coords = torch_scatter.segment_csr(
                gates.unsqueeze(-1) * point.coord.float().unsqueeze(1),
                segment_offsets,
                reduce="sum",
            ) / latent_mass.unsqueeze(-1)
        else:
            latent_parts = []
            latent_coord_parts = []
            for batch_index in range(int(point.offset.numel())):
                start = 0 if batch_index == 0 else int(point.offset[batch_index - 1].item())
                stop = int(point.offset[batch_index].item())
                batch_scores = scores[start:stop].softmax(dim=0)
                if self.voxel_density_power > 0.0:
                    # Preserve PTv3's learned normalized readout while exposing
                    # local sampling density to the operator.  Counts are raw
                    # points represented by each final serialized voxel; the
                    # per-batch mean keeps the overall latent scale stable.
                    batch_density = point.voxel_count[start:stop].float().clamp_min(1.0)
                    batch_density = batch_density / batch_density.mean().clamp_min(1.0)
                    density_factor = batch_density.pow(self.voxel_density_power).unsqueeze(-1)
                    batch_scores = batch_scores * density_factor
                    batch_scores = batch_scores / batch_scores.sum(dim=0, keepdim=True).clamp_min(1.0e-8)
                latent_parts.append(torch.einsum("nl,nd->ld", batch_scores, values[start:stop]))
                latent_coord_parts.append(torch.einsum("nl,nc->lc", batch_scores, point.coord[start:stop].float()))
            latent = torch.stack(latent_parts, dim=0).float()
            latent_coords = torch.stack(latent_coord_parts, dim=0).float()
        # Normalize only feature channels after the density-preserving sum;
        # the point-mass-dependent content has already been retained.
        latent = latent.float()
        latent_coords = latent_coords.float()
        with torch.autocast(device_type=device_type, enabled=False):
            latent = self.geometry_cond(self.latent_norm(latent), params.float() if params is not None else None)
        return latent, latent_coords, point.coord, point.feat, point.batch

    def decode_features(
        self,
        latent,
        surf_query_pos,
        vol_query_pos,
        params=None,
        latent_coords=None,
        geometry_coords=None,
        geometry_features=None,
        geometry_batch=None,
    ):
        surf_count = int(surf_query_pos.shape[1])
        queries = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        if queries.shape[1] == 0:
            empty = queries.new_empty((queries.shape[0], 0, self.surface_channels + self.volume_channels))
            return split_surface_volume_predictions(empty, surf_query_pos, self.surface_channels)
        # The decoder is pointwise over up to 131k queries, so keeping it in
        # FP32 is affordable relative to the sparse backbone and is important:
        # q @ k^T can exceed the FP16 range long before the final prediction
        # does.  Casting q/k to float32 alone is insufficient while autocast is
        # active because matmul would still be selected as an FP16 kernel.
        device_type = queries.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            query = self.query_embed(queries.float() * self.pos_scale_factor)
            if (
                self.use_local_query_features
                and geometry_coords is not None
                and geometry_features is not None
                and geometry_batch is not None
            ):
                local_features = _query_local_geometry_features(
                    queries,
                    geometry_coords,
                    geometry_features,
                    geometry_batch,
                    neighbors=self.local_query_neighbors,
                    chunk_size=self.local_query_chunk_size,
                ).float()
                query = query + self.local_geometry_proj(local_features)
            type_ids = torch.cat(
                [
                    torch.zeros(surf_count, device=queries.device, dtype=torch.long),
                    torch.ones(int(vol_query_pos.shape[1]), device=queries.device, dtype=torch.long),
                ],
                dim=0,
            )
            query = query + self.query_type.index_select(0, type_ids).unsqueeze(0)
            query = self.query_cond(query, params.float() if params is not None else None)
            q = self.query_q(self.query_norm(query))
            if latent_coords is not None:
                # Use the existing coordinate encoder so keys and queries share
                # a positional language without materializing a query-by-geometry
                # attention matrix.
                latent = latent.float() + self.query_embed(latent_coords.float() * self.pos_scale_factor)
            else:
                latent = latent.float()
            k = self.latent_k(self.latent_attention_norm(latent))
            v = self.latent_v(latent)
            attention = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            decoded = query + attention.softmax(dim=-1) @ v
            decoded = decoded + self.query_mlp(self.query_norm(decoded))
            pred = self.output_head(decoded)
        return split_surface_volume_predictions(pred, surf_query_pos, self.surface_channels)

    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None, return_latent=False):
        del geo_log_density
        latent, latent_coords, geometry_coords, geometry_features, geometry_batch = self.encode_geometry(
            geo,
            params=params,
        )
        pred_surf, pred_vol = self.decode_features(
            latent,
            surf_query_pos,
            vol_query_pos,
            params=params,
            latent_coords=latent_coords,
            geometry_coords=geometry_coords,
            geometry_features=geometry_features,
            geometry_batch=geometry_batch,
        )
        if return_latent:
            return pred_surf, pred_vol, latent
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        del geo_log_density
        latent, latent_coords, geometry_coords, geometry_features, geometry_batch = self.encode_geometry(
            geo,
            params=params,
        )
        total_queries = int(surf_query_pos.shape[1] + vol_query_pos.shape[1])
        if total_queries <= self.query_chunk_size:
            return self.decode_features(
                latent,
                surf_query_pos,
                vol_query_pos,
                params=params,
                latent_coords=latent_coords,
                geometry_coords=geometry_coords,
                geometry_features=geometry_features,
                geometry_batch=geometry_batch,
            )
        surf_count = int(surf_query_pos.shape[1])
        surf_parts, vol_parts = [], []
        full_query = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        for start in range(0, total_queries, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, total_queries)
            query = full_query[:, start:stop]
            local_surface = max(0, min(stop, surf_count) - start)
            surf_chunk = query[:, :local_surface]
            vol_chunk = query[:, local_surface:]
            if surf_chunk.shape[1] > 0:
                surf_pred, _ = self.decode_features(
                    latent,
                    surf_chunk,
                    surf_chunk[:, :0],
                    params=params,
                    latent_coords=latent_coords,
                    geometry_coords=geometry_coords,
                    geometry_features=geometry_features,
                    geometry_batch=geometry_batch,
                )
                surf_parts.append(surf_pred)
            if vol_chunk.shape[1] > 0:
                _, vol_pred = self.decode_features(
                    latent,
                    vol_chunk[:, :0],
                    vol_chunk,
                    params=params,
                    latent_coords=latent_coords,
                    geometry_coords=geometry_coords,
                    geometry_features=geometry_features,
                    geometry_batch=geometry_batch,
                )
                vol_parts.append(vol_pred)
        surf_pred = torch.cat(surf_parts, dim=1) if surf_parts else surf_query_pos.new_empty((surf_query_pos.shape[0], 0, self.surface_channels))
        vol_pred = torch.cat(vol_parts, dim=1) if vol_parts else vol_query_pos.new_empty((vol_query_pos.shape[0], 0, self.volume_channels))
        return surf_pred, vol_pred


class PointTransformerV3WithLatent(PointTransformerV3):
    pass
