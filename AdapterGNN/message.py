import inspect
import os
import os.path as osp
import random
import re
from collections import OrderedDict
from inspect import Parameter
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
    get_type_hints,
)

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from torch_geometric.nn.aggr import Aggregation
try:
    from torch_geometric.inspector import Inspector, Signature
except Exception:  # PyG <= 2.3-ish
    from torch_geometric.nn.conv.utils.inspector import Inspector  # no Signature back then

# These old templating utilities are no longer needed; kept only for back-compat if you call them elsewhere.
try:
    from torch_geometric.template import module_from_template
except Exception:  # old PyG
    from torch_geometric.nn.conv.utils.jit import class_from_module_repr

# Typing utils existed only on very old PyG; we won't use them anymore.
try:
    from torch_geometric.nn.conv.utils.typing import (
        parse_types, resolve_types, sanitize, split_types_repr
    )
except Exception:
    parse_types = resolve_types = sanitize = split_types_repr = None

from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.typing import Adj, Size, SparseTensor
from torch_geometric.utils import (
    is_sparse,
    is_torch_sparse_tensor,
    to_edge_index,
)

# ptr2index canonical import moved; keep a fallback for older builds
try:
    from torch_geometric.index import ptr2index
except Exception:
    from torch_geometric.utils.sparse import ptr2index

FUSE_AGGRS = {'add', 'sum', 'mean', 'min', 'max'}


def ptr2ind(ptr: Tensor) -> Tensor:
    # Unused helper; retained in case external code references it.
    ind = torch.arange(ptr.numel() - 1, device=ptr.device)
    return ind.repeat_interleave(ptr[1:] - ptr[:-1])


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers of the form

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \bigoplus_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),

    where :math:`\bigoplus` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean, min, max or mul, and
    :math:`\gamma_{\mathbf{\Theta}}` and :math:`\phi_{\mathbf{\Theta}}` denote
    differentiable functions such as MLPs.
    """

    special_args: Set[str] = {
        'edge_index', 'adj_t', 'edge_index_i', 'edge_index_j', 'size',
        'size_i', 'size_j', 'ptr', 'index', 'dim_size'
    }

    def __init__(
        self,
        aggr: Optional[Union[str, List[str], Aggregation]] = "add",
        *,
        aggr_kwargs: Optional[Dict[str, Any]] = None,
        flow: str = "source_to_target",
        node_dim: int = -2,
        decomposed_layers: int = 1,
        **kwargs,
    ):
        super().__init__()

        if aggr is None:
            self.aggr = None
        elif isinstance(aggr, (str, Aggregation)):
            self.aggr = str(aggr)
        elif isinstance(aggr, (tuple, list)):
            self.aggr = [str(x) for x in aggr]

        self.aggr_module = aggr_resolver(aggr, **(aggr_kwargs or {}))

        self.flow = flow

        if flow not in ['source_to_target', 'target_to_source']:
            raise ValueError(f"Expected 'flow' to be either 'source_to_target'"
                             f" or 'target_to_source' (got '{flow}')")

        self.node_dim = node_dim
        self.decomposed_layers = decomposed_layers

        # ----- Inspector setup (PyG 2.7 compatible) -----
        self.inspector = Inspector(self.__class__)
        # Record signatures; use excludes to drop self or legacy 'aggr' args:
        self.inspector.inspect_signature(self.message)
        self.inspector.inspect_signature(self.aggregate, exclude=[0, 'aggr'])
        self.inspector.inspect_signature(self.message_and_aggregate, [0])  # exclude first positional (self/edge_index)
        self.inspector.inspect_signature(self.update, exclude=[0])
        self.inspector.inspect_signature(self.edge_update)

        # Build param sets using flat name helpers and excludes:
        self._user_args = set(self.inspector.get_flat_param_names(
            ['message', 'aggregate', 'update'], exclude=self.special_args))
        self._fused_user_args = set(self.inspector.get_flat_param_names(
            ['message_and_aggregate', 'update'], exclude=self.special_args))
        self._edge_user_args = set(self.inspector.get_param_names(
            'edge_update', exclude=self.special_args))

        # Support for "fused" message passing.
        self.fuse = self.inspector.implements('message_and_aggregate')
        if self.aggr is not None:
            self.fuse &= isinstance(self.aggr, str) and self.aggr in FUSE_AGGRS

        # Support for explainability.
        self._explain = False
        self._edge_mask = None
        self._loop_mask = None
        self._apply_sigmoid = True

        # Hooks:
        self._propagate_forward_pre_hooks = OrderedDict()
        self._propagate_forward_hooks = OrderedDict()
        self._message_forward_pre_hooks = OrderedDict()
        self._message_forward_hooks = OrderedDict()
        self._aggregate_forward_pre_hooks = OrderedDict()
        self._aggregate_forward_hooks = OrderedDict()
        self._message_and_aggregate_forward_pre_hooks = OrderedDict()
        self._message_and_aggregate_forward_hooks = OrderedDict()
        self._edge_update_forward_pre_hooks = OrderedDict()
        self._edge_update_forward_hooks = OrderedDict()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.aggr_module is not None:
            self.aggr_module.reset_parameters()

    def forward(self, *args, **kwargs) -> Any:
        r"""Runs the forward pass of the module."""
        pass

    def _check_input(self, edge_index, size):
        the_size: List[Optional[int]] = [None, None]

        if is_sparse(edge_index):
            if self.flow == 'target_to_source':
                raise ValueError(
                    ('Flow direction "target_to_source" is invalid for '
                     'message propagation via `torch_sparse.SparseTensor` '
                     'or `torch.sparse.Tensor`. If you really want to make '
                     'use of a reverse message passing flow, pass in the '
                     'transposed sparse tensor to the message passing module, '
                     'e.g., `adj_t.t()`.'))
            the_size[0] = edge_index.size(1)
            the_size[1] = edge_index.size(0)
            return the_size
        elif isinstance(edge_index, Tensor):
            int_dtypes = (torch.uint8, torch.int8, torch.int32, torch.int64)

            if edge_index.dtype not in int_dtypes:
                raise ValueError(f"Expected 'edge_index' to be of integer "
                                 f"type (got '{edge_index.dtype}')")
            if edge_index.dim() != 2:
                raise ValueError(f"Expected 'edge_index' to be two-dimensional"
                                 f" (got {edge_index.dim()} dimensions)")
            if not torch.jit.is_tracing() and edge_index.size(0) != 2:
                raise ValueError(f"Expected 'edge_index' to have size '2' in "
                                 f"the first dimension (got "
                                 f"'{edge_index.size(0)}')")
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size

        raise ValueError(
            ('`MessagePassing.propagate` only supports integer tensors of '
             'shape `[2, num_messages]`, `torch_sparse.SparseTensor` or '
             '`torch.sparse.Tensor` for argument `edge_index`.'))

    def _set_size(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                (f'Encountered tensor with size {src.size(self.node_dim)} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))

    def _lift(self, src, edge_index, dim):
        if is_torch_sparse_tensor(edge_index):
            assert dim == 0 or dim == 1
            if edge_index.layout == torch.sparse_coo:
                index = edge_index._indices()[1 - dim]
            elif edge_index.layout == torch.sparse_csr:
                if dim == 0:
                    index = edge_index.col_indices()
                else:
                    index = ptr2index(edge_index.crow_indices())
            elif edge_index.layout == torch.sparse_csc:
                if dim == 0:
                    index = ptr2index(edge_index.ccol_indices())
                else:
                    index = edge_index.row_indices()
            else:
                raise ValueError(f"Unsupported sparse tensor layout "
                                 f"(got '{edge_index.layout}')")
            return src.index_select(self.node_dim, index)

        elif isinstance(edge_index, Tensor):
            try:
                index = edge_index[dim]
                return src.index_select(self.node_dim, index)
            except (IndexError, RuntimeError) as e:
                if index.min() < 0 or index.max() >= src.size(self.node_dim):
                    raise IndexError(
                        f"Encountered an index error. Please ensure that all "
                        f"indices in 'edge_index' point to valid indices in "
                        f"the interval [0, {src.size(self.node_dim) - 1}] "
                        f"(got interval "
                        f"[{int(index.min())}, {int(index.max())}])")
                else:
                    raise e

                if index.numel() > 0 and index.min() < 0:
                    raise ValueError(
                        f"Found negative indices in 'edge_index' (got "
                        f"{index.min().item()}). Please ensure that all "
                        f"indices in 'edge_index' point to valid indices "
                        f"in the interval [0, {src.size(self.node_dim)}) in "
                        f"your node feature matrix and try again.")

                if (index.numel() > 0
                        and index.max() >= src.size(self.node_dim)):
                    raise ValueError(
                        f"Found indices in 'edge_index' that are larger "
                        f"than {src.size(self.node_dim) - 1} (got "
                        f"{index.max().item()}). Please ensure that all "
                        f"indices in 'edge_index' point to valid indices "
                        f"in the interval [0, {src.size(self.node_dim)}) in "
                        f"your node feature matrix and try again.")

                raise e

        elif isinstance(edge_index, SparseTensor):
            row, col, _ = edge_index.coo()
            if dim == 0:
                return src.index_select(self.node_dim, col)
            elif dim == 1:
                return src.index_select(self.node_dim, row)

        raise ValueError(
            ('`MessagePassing.propagate` only supports integer tensors of '
             'shape `[2, num_messages]`, `torch_sparse.SparseTensor` '
             'or `torch.sparse.Tensor` for argument `edge_index`.'))

    def _collect(self, args, edge_index, size, kwargs):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        out = {}
        for arg in args:
            if arg[-2:] not in ['_i', '_j']:
                out[arg] = kwargs.get(arg, Parameter.empty)
            else:
                dim = j if arg[-2:] == '_j' else i
                data = kwargs.get(arg[:-2], Parameter.empty)

                if isinstance(data, (tuple, list)):
                    assert len(data) == 2
                    if isinstance(data[1 - dim], Tensor):
                        self._set_size(size, 1 - dim, data[1 - dim])
                    data = data[dim]

                if isinstance(data, Tensor):
                    self._set_size(size, dim, data)
                    data = self._lift(data, edge_index, dim)

                out[arg] = data

        if is_torch_sparse_tensor(edge_index):
            indices, values = to_edge_index(edge_index)
            out['adj_t'] = edge_index
            out['edge_index'] = None
            out['edge_index_i'] = indices[0]
            out['edge_index_j'] = indices[1]
            out['ptr'] = None  # TODO Get `rowptr` from CSR representation.
            if out.get('edge_weight', None) is None:
                out['edge_weight'] = values
            if out.get('edge_attr', None) is None:
                out['edge_attr'] = None if values.dim() == 1 else values
            if out.get('edge_type', None) is None:
                out['edge_type'] = values

        elif isinstance(edge_index, Tensor):
            out['adj_t'] = None
            out['edge_index'] = edge_index
            out['edge_index_i'] = edge_index[i]
            out['edge_index_j'] = edge_index[j]
            out['ptr'] = None

        elif isinstance(edge_index, SparseTensor):
            row, col, value = edge_index.coo()
            rowptr, _, _ = edge_index.csr()

            out['adj_t'] = edge_index
            out['edge_index'] = None
            out['edge_index_i'] = row
            out['edge_index_j'] = col
            out['ptr'] = rowptr
            if out.get('edge_weight', None) is None:
                out['edge_weight'] = value
            if out.get('edge_attr', None) is None:
                out['edge_attr'] = value
            if out.get('edge_type', None) is None:
                out['edge_type'] = value

        out['index'] = out['edge_index_i']
        out['size'] = size
        out['size_i'] = size[i] if size[i] is not None else size[j]
        out['size_j'] = size[j] if size[j] is not None else size[i]
        out['dim_size'] = out['size_i']

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages."""
        decomposed_layers = 1 if self.explain else self.decomposed_layers

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        size = self._check_input(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if is_sparse(edge_index) and self.fuse and not self.explain:
            coll_dict = self._collect(self._fused_user_args, edge_index, size,
                                      kwargs)

            msg_aggr_kwargs = self.inspector.collect_param_data(
                'message_and_aggregate', coll_dict)
            for hook in self._message_and_aggregate_forward_pre_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs))
                if res is not None:
                    edge_index, msg_aggr_kwargs = res
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
            for hook in self._message_and_aggregate_forward_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs), out)
                if res is not None:
                    out = res

            update_kwargs = self.inspector.collect_param_data('update', coll_dict)
            out = self.update(out, **update_kwargs)

        else:  # Otherwise, run both functions in separation.
            if decomposed_layers > 1:
                user_args = self._user_args
                decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
                decomp_kwargs = {
                    a: kwargs[a].chunk(decomposed_layers, -1)
                    for a in decomp_args
                }
                decomp_out = []

            for i in range(decomposed_layers):
                if decomposed_layers > 1:
                    for arg in decomp_args:
                        kwargs[arg] = decomp_kwargs[arg][i]

                coll_dict = self._collect(self._user_args, edge_index, size,
                                          kwargs)

                msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
                for hook in self._message_forward_pre_hooks.values():
                    res = hook(self, (msg_kwargs, ))
                    if res is not None:
                        msg_kwargs = res[0] if isinstance(res, tuple) else res
                out = self.message(**msg_kwargs)
                for hook in self._message_forward_hooks.values():
                    res = hook(self, (msg_kwargs, ), out)
                    if res is not None:
                        out = res

                if self.explain:
                    explain_msg_kwargs = self.inspector.collect_param_data(
                        'explain_message', coll_dict)
                    out = self.explain_message(out, **explain_msg_kwargs)

                aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
                for hook in self._aggregate_forward_pre_hooks.values():
                    res = hook(self, (aggr_kwargs, ))
                    if res is not None:
                        aggr_kwargs = res[0] if isinstance(res, tuple) else res

                out = self.aggregate(out, **aggr_kwargs)

                for hook in self._aggregate_forward_hooks.values():
                    res = hook(self, (aggr_kwargs, ), out)
                    if res is not None:
                        out = res

                update_kwargs = self.inspector.collect_param_data('update', coll_dict)
                out = self.update(out, **update_kwargs)

                if decomposed_layers > 1:
                    decomp_out.append(out)

            if decomposed_layers > 1:
                out = torch.cat(decomp_out, dim=-1)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out

    def edge_updater(self, edge_index: Adj, **kwargs):
        r"""The initial call to compute or update features for each edge."""
        for hook in self._edge_update_forward_pre_hooks.values():
            res = hook(self, (edge_index, kwargs))
            if res is not None:
                edge_index, kwargs = res

        size = self._check_input(edge_index, size=None)

        coll_dict = self._collect(self._edge_user_args, edge_index, size,
                                  kwargs)

        edge_kwargs = self.inspector.collect_param_data('edge_update', coll_dict)
        out = self.edge_update(**edge_kwargs)

        for hook in self._edge_update_forward_hooks.values():
            res = hook(self, (edge_index, kwargs), out)
            if res is not None:
                out = res

        return out

    def message(self, x_j: Tensor) -> Tensor:
        r"""Constructs messages from node j to node i."""
        return x_j

    @property
    def explain(self) -> bool:
        return self._explain

    @explain.setter
    def explain(self, explain: bool):
        if explain:
            methods = ['message', 'explain_message', 'aggregate', 'update']
        else:
            methods = ['message', 'aggregate', 'update']

        self._explain = explain
        self.inspector.inspect(self.explain_message)
        self._user_args = self.inspector.keys(methods).difference(
            self.special_args)

    def explain_message(self, inputs: Tensor, size_i: int) -> Tensor:
        edge_mask = self._edge_mask

        if edge_mask is None:
            raise ValueError(f"Could not find a pre-defined 'edge_mask' as "
                             f"part of {self.__class__.__name__}.")

        if self._apply_sigmoid:
            edge_mask = edge_mask.sigmoid()

        if inputs.size(self.node_dim) != edge_mask.size(0):
            edge_mask = edge_mask[self._loop_mask]
            loop = edge_mask.new_ones(size_i)
            edge_mask = torch.cat([edge_mask, loop], dim=0)
        assert inputs.size(self.node_dim) == edge_mask.size(0)

        size = [1] * inputs.dim()
        size[self.node_dim] = -1
        return inputs * edge_mask.view(size)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size,
                                dim=self.node_dim)

    def message_and_aggregate(
        self,
        adj_t: Union[SparseTensor, Tensor],
    ) -> Tensor:
        raise NotImplementedError

    def update(self, inputs: Tensor) -> Tensor:
        return inputs

    def edge_update(self) -> Tensor:
        raise NotImplementedError

    def register_propagate_forward_pre_hook(self,
                                            hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._propagate_forward_pre_hooks)
        self._propagate_forward_pre_hooks[handle.id] = hook
        return handle

    def register_propagate_forward_hook(self,
                                        hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._propagate_forward_hooks)
        self._propagate_forward_hooks[handle.id] = hook
        return handle

    def register_message_forward_pre_hook(self,
                                          hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._message_forward_pre_hooks)
        self._message_forward_pre_hooks[handle.id] = hook
        return handle

    def register_message_forward_hook(self, hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._message_forward_hooks)
        self._message_forward_hooks[handle.id] = hook
        return handle

    def register_aggregate_forward_pre_hook(self,
                                            hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._aggregate_forward_pre_hooks)
        self._aggregate_forward_pre_hooks[handle.id] = hook
        return handle

    def register_aggregate_forward_hook(self,
                                        hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._aggregate_forward_hooks)
        self._aggregate_forward_hooks[handle.id] = hook
        return handle

    def register_message_and_aggregate_forward_pre_hook(
            self, hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._message_and_aggregate_forward_pre_hooks)
        self._message_and_aggregate_forward_pre_hooks[handle.id] = hook
        return handle

    def register_message_and_aggregate_forward_hook(
            self, hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._message_and_aggregate_forward_hooks)
        self._message_and_aggregate_forward_hooks[handle.id] = hook
        return handle

    def register_edge_update_forward_pre_hook(
            self, hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._edge_update_forward_pre_hooks)
        self._edge_update_forward_pre_hooks[handle.id] = hook
        return handle

    def register_edge_update_forward_hook(self,
                                          hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._edge_update_forward_hooks)
        self._edge_update_forward_hooks[handle.id] = hook
        return handle

    # PyG ≥ 2.5: jittable is deprecated/no-op; keep the method for API compatibility.
    def jittable(self, typing: Optional[str] = None) -> 'MessagePassing':
        import warnings
        warnings.warn(
            f"'{self.__class__.__name__}.jittable' is deprecated and a no-op.",
            stacklevel=2
        )
        return self

    def __repr__(self) -> str:
        if hasattr(self, 'in_channels') and hasattr(self, 'out_channels'):
            return (f'{self.__class__.__name__}({self.in_channels}, '
                    f'{self.out_channels})')
        return f'{self.__class__.__name__}()'
