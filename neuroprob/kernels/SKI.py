#!/usr/bin/env python3

from typing import List, Optional, Tuple, Union

import torch

from ..lazy import InterpolatedLazyTensor, lazify
from ..models.exact_prediction_strategies import InterpolatedPredictionStrategy
from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.grid import create_grid
from ..utils.interpolation import Interpolation
#from .grid_kernel import GridKernel
from .kernel import Kernel






#!/usr/bin/env python3

from typing import Optional

import torch
from torch import Tensor

from .. import settings
from ..lazy import KroneckerProductLazyTensor, ToeplitzLazyTensor, delazify
from ..utils.grid import convert_legacy_grid, create_data_from_grid
from .kernel import Kernel


class GridKernel(Kernel):
    r"""
    If the input data :math:`X` are regularly spaced on a grid, then
    `GridKernel` can dramatically speed up computatations for stationary kernel.
    GridKernel exploits Toeplitz and Kronecker structure within the covariance matrix.
    See `Fast kernel learning for multidimensional pattern extrapolation`_ for more info.
    .. note::
        `GridKernel` can only wrap **stationary kernels** (such as RBF, Matern,
        Periodic, Spectral Mixture, etc.)
    Args:
        :attr:`base_kernel` (Kernel):
            The kernel to speed up with grid methods.
        :attr:`grid` (Tensor):
            A g x d tensor where column i consists of the projections of the
            grid in dimension i.
        :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel`.
        :attr:`interpolation_mode` (bool):
            Used for GridInterpolationKernel where we want the covariance
            between points in the projections of the grid of each dimension.
            We do this by treating `grid` as d batches of g x 1 tensors by
            calling base_kernel(grid, grid) with last_dim_is_batch to get a d x g x g Tensor
            which we Kronecker product to get a g x g KroneckerProductLazyTensor.
    .. _Fast kernel learning for multidimensional pattern extrapolation:
        http://www.cs.cmu.edu/~andrewgw/manet.pdf
    """

    is_stationary = True

    def __init__(
        self,
        base_kernel: Kernel,
        grid: Tensor,
        interpolation_mode: Optional[bool] = False,
        active_dims: Optional[bool] = None,
    ):
        if not base_kernel.is_stationary:
            raise RuntimeError("The base_kernel for GridKernel must be stationary.")

        super().__init__(active_dims=active_dims)
        if torch.is_tensor(grid):
            grid = convert_legacy_grid(grid)
        self.interpolation_mode = interpolation_mode
        self.base_kernel = base_kernel
        self.num_dims = len(grid)
        self.register_buffer_list("grid", grid)
        if not self.interpolation_mode:
            self.register_buffer("full_grid", create_data_from_grid(grid))

    def _clear_cache(self):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat

    def register_buffer_list(self, base_name, tensors):
        """Helper to register several buffers at once under a single base name"""
        for i, tensor in enumerate(tensors):
            self.register_buffer(base_name + "_" + str(i), tensor)

    @property
    def grid(self):
        return [getattr(self, f"grid_{i}") for i in range(self.num_dims)]

    def update_grid(self, grid):
        """
        Supply a new `grid` if it ever changes.
        """
        if torch.is_tensor(grid):
            grid = convert_legacy_grid(grid)

        if len(grid) != self.num_dims:
            raise RuntimeError("New grid should have the same number of dimensions as before.")

        for i in range(self.num_dims):
            setattr(self, f"grid_{i}", grid[i])

        if not self.interpolation_mode:
            self.full_grid = create_data_from_grid(self.grid)

        self._clear_cache()
        return self

    @property
    def is_ragged(self):
        return not all(self.grid[0].size() == proj.size() for proj in self.grid)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch and not self.interpolation_mode:
            raise ValueError("last_dim_is_batch is only valid with interpolation model")

        grid = self.grid
        if self.is_ragged:
            # Pad the grid - so that grid is the same size for each dimension
            max_grid_size = max(proj.size(-1) for proj in grid)
            padded_grid = []
            for proj in grid:
                padding_size = max_grid_size - proj.size(-1)
                if padding_size > 0:
                    dtype = proj.dtype
                    device = proj.device
                    padded_grid.append(
                        torch.cat([proj, torch.zeros(*proj.shape[:-1], padding_size, dtype=dtype, device=device)])
                    )
                else:
                    padded_grid.append(proj)
        else:
            padded_grid = grid

        if not self.interpolation_mode:
            if len(x1.shape[:-2]):
                full_grid = self.full_grid.expand(*x1.shape[:-2], *self.full_grid.shape[-2:])
            else:
                full_grid = self.full_grid

        if self.interpolation_mode or (torch.equal(x1, full_grid) and torch.equal(x2, full_grid)):
            if not self.training and hasattr(self, "_cached_kernel_mat"):
                return self._cached_kernel_mat
            # Can exploit Toeplitz structure if grid points in each dimension are equally
            # spaced and using a translation-invariant kernel
            if settings.use_toeplitz.on():
                # Use padded grid for batch mode
                first_grid_point = torch.stack([proj[0].unsqueeze(0) for proj in grid], dim=-1)
                full_grid = torch.stack(padded_grid, dim=-1)
                covars = delazify(self.base_kernel(first_grid_point, full_grid, last_dim_is_batch=True, **params))

                if last_dim_is_batch:
                    # Toeplitz expects batches of columns so we concatenate the
                    # 1 x grid_size[i] tensors together
                    # Note that this requires all the dimensions to have the same number of grid points
                    covar = ToeplitzLazyTensor(covars.squeeze(-2))
                else:
                    # Non-batched ToeplitzLazyTensor expects a 1D tensor, so we squeeze out the row dimension
                    covars = covars.squeeze(-2)  # Get rid of the dimension corresponding to the first point
                    # Un-pad the grid
                    covars = [ToeplitzLazyTensor(covars[..., i, : proj.size(-1)]) for i, proj in enumerate(grid)]
                    # Due to legacy reasons, KroneckerProductLazyTensor(A, B, C) is actually (C Kron B Kron A)
                    covar = KroneckerProductLazyTensor(*covars[::-1])
            else:
                full_grid = torch.stack(padded_grid, dim=-1)
                covars = delazify(self.base_kernel(full_grid, full_grid, last_dim_is_batch=True, **params))
                if last_dim_is_batch:
                    # Note that this requires all the dimensions to have the same number of grid points
                    covar = covars
                else:
                    covars = [covars[..., i, : proj.size(-1), : proj.size(-1)] for i, proj in enumerate(self.grid)]
                    covar = KroneckerProductLazyTensor(*covars[::-1])

            if not self.training:
                self._cached_kernel_mat = covar

            return covar
        else:
            return self.base_kernel.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)












class GridInterpolationKernel(GridKernel):
    r"""
    Implements the KISS-GP (or SKI) approximation for a given kernel.
    It was proposed in `Kernel Interpolation for Scalable Structured Gaussian Processes`_,
    and offers extremely fast and accurate Kernel approximations for large datasets.

    Given a base kernel `k`, the covariance :math:`k(\mathbf{x_1}, \mathbf{x_2})` is approximated by
    using a grid of regularly spaced *inducing points*:

    .. math::

       \begin{equation*}
          k(\mathbf{x_1}, \mathbf{x_2}) = \mathbf{w_{x_1}}^\top K_{U,U} \mathbf{w_{x_2}}
       \end{equation*}

    where

    * :math:`U` is the set of gridded inducing points

    * :math:`K_{U,U}` is the kernel matrix between the inducing points

    * :math:`\mathbf{w_{x_1}}` and :math:`\mathbf{w_{x_2}}` are sparse vectors based on
      :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}` that apply cubic interpolation.

    The user should supply the size of the grid (using the :attr:`grid_size` attribute).
    To choose a reasonable grid value, we highly recommend using the
    :func:`gpytorch.utils.grid.choose_grid_size` helper function.
    The bounds of the grid will automatically be determined by data.

    (Alternatively, you can hard-code bounds using the :attr:`grid_bounds`, which
    will speed up this kernel's computations.)

    .. note::

        `GridInterpolationKernel` can only wrap **stationary kernels** (such as RBF, Matern,
        Periodic, Spectral Mixture, etc.)

    Args:
        - :attr:`base_kernel` (Kernel):
            The kernel to approximate with KISS-GP
        - :attr:`grid_size` (Union[int, List[int]]):
            The size of the grid in each dimension.
            If a single int is provided, then every dimension will have the same grid size.
        - :attr:`num_dims` (int):
            The dimension of the input data. Required if `grid_bounds=None`
        - :attr:`grid_bounds` (tuple(float, float), optional):
            The bounds of the grid, if known (high performance mode).
            The length of the tuple must match the number of dimensions.
            The entries represent the min/max values for each dimension.
        - :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel`.

    .. _Kernel Interpolation for Scalable Structured Gaussian Processes:
        http://proceedings.mlr.press/v37/wilson15.pdf
    """

    def __init__(
        self,
        base_kernel: Kernel,
        grid_size: Union[int, List[int]],
        num_dims: int = None,
        grid_bounds: Optional[Tuple[float, float]] = None,
        active_dims: Tuple[int, ...] = None,
    ):
        has_initialized_grid = 0
        grid_is_dynamic = True

        # Make some temporary grid bounds, if none exist
        if grid_bounds is None:
            if num_dims is None:
                raise RuntimeError("num_dims must be supplied if grid_bounds is None")
            else:
                # Create some temporary grid bounds - they'll be changed soon
                grid_bounds = tuple((-1.0, 1.0) for _ in range(num_dims))
        else:
            has_initialized_grid = 1
            grid_is_dynamic = False
            if num_dims is None:
                num_dims = len(grid_bounds)
            elif num_dims != len(grid_bounds):
                raise RuntimeError(
                    "num_dims ({}) disagrees with the number of supplied "
                    "grid_bounds ({})".format(num_dims, len(grid_bounds))
                )

        if isinstance(grid_size, int):
            grid_sizes = [grid_size for _ in range(num_dims)]
        else:
            grid_sizes = list(grid_size)

        if len(grid_sizes) != num_dims:
            raise RuntimeError("The number of grid sizes provided through grid_size do not match num_dims.")

        # Initialize values and the grid
        self.grid_is_dynamic = grid_is_dynamic
        self.num_dims = num_dims
        self.grid_sizes = grid_sizes
        self.grid_bounds = grid_bounds
        grid = create_grid(self.grid_sizes, self.grid_bounds)

        super(GridInterpolationKernel, self).__init__(
            base_kernel=base_kernel, grid=grid, interpolation_mode=True, active_dims=active_dims,
        )
        self.register_buffer("has_initialized_grid", torch.tensor(has_initialized_grid, dtype=torch.bool))

    @property
    def _tight_grid_bounds(self):
        grid_spacings = tuple((bound[1] - bound[0]) / self.grid_sizes[i] for i, bound in enumerate(self.grid_bounds))
        return tuple(
            (bound[0] + 2.01 * spacing, bound[1] - 2.01 * spacing)
            for bound, spacing in zip(self.grid_bounds, grid_spacings)
        )

    def _compute_grid(self, inputs, last_dim_is_batch=False):
        n_data, n_dimensions = inputs.size(-2), inputs.size(-1)
        if last_dim_is_batch:
            inputs = inputs.transpose(-1, -2).unsqueeze(-1)
            n_dimensions = 1
        batch_shape = inputs.shape[:-2]

        inputs = inputs.reshape(-1, n_dimensions)
        interp_indices, interp_values = Interpolation().interpolate(self.grid, inputs)
        interp_indices = interp_indices.view(*batch_shape, n_data, -1)
        interp_values = interp_values.view(*batch_shape, n_data, -1)
        return interp_indices, interp_values

    def _inducing_forward(self, last_dim_is_batch, **params):
        return super().forward(self.grid, self.grid, last_dim_is_batch=last_dim_is_batch, **params)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # See if we need to update the grid or not
        if self.grid_is_dynamic:  # This is true if a grid_bounds wasn't passed in
            if torch.equal(x1, x2):
                x = x1.reshape(-1, self.num_dims)
            else:
                x = torch.cat([x1.reshape(-1, self.num_dims), x2.reshape(-1, self.num_dims)])
            x_maxs = x.max(0)[0].tolist()
            x_mins = x.min(0)[0].tolist()

            # We need to update the grid if
            # 1) it hasn't ever been initialized, or
            # 2) if any of the grid points are "out of bounds"
            update_grid = (not self.has_initialized_grid.item()) or any(
                x_min < bound[0] or x_max > bound[1]
                for x_min, x_max, bound in zip(x_mins, x_maxs, self._tight_grid_bounds)
            )

            # Update the grid if needed
            if update_grid:
                grid_spacings = tuple(
                    (x_max - x_min) / (gs - 4.02) for gs, x_min, x_max in zip(self.grid_sizes, x_mins, x_maxs)
                )
                self.grid_bounds = tuple(
                    (x_min - 2.01 * spacing, x_max + 2.01 * spacing)
                    for x_min, x_max, spacing in zip(x_mins, x_maxs, grid_spacings)
                )
                grid = create_grid(
                    self.grid_sizes, self.grid_bounds, dtype=self.grid[0].dtype, device=self.grid[0].device,
                )
                self.update_grid(grid)

        base_lazy_tsr = lazify(self._inducing_forward(last_dim_is_batch=last_dim_is_batch, **params))
        if last_dim_is_batch:
            base_lazy_tsr = base_lazy_tsr.repeat(*x1.shape[:-2], x1.size(-1), 1, 1)

        left_interp_indices, left_interp_values = self._compute_grid(x1, last_dim_is_batch)
        if torch.equal(x1, x2):
            right_interp_indices = left_interp_indices
            right_interp_values = left_interp_values
        else:
            right_interp_indices, right_interp_values = self._compute_grid(x2, last_dim_is_batch)

        batch_shape = _mul_broadcast_shape(
            base_lazy_tsr.batch_shape, left_interp_indices.shape[:-2], right_interp_indices.shape[:-2],
        )
        res = InterpolatedLazyTensor(
            base_lazy_tsr.expand(*batch_shape, *base_lazy_tsr.matrix_shape),
            left_interp_indices.detach().expand(*batch_shape, *left_interp_indices.shape[-2:]),
            left_interp_values.expand(*batch_shape, *left_interp_values.shape[-2:]),
            right_interp_indices.detach().expand(*batch_shape, *right_interp_indices.shape[-2:]),
            right_interp_values.expand(*batch_shape, *right_interp_values.shape[-2:]),
        )

        if diag:
            return res.diag()
        else:
            return res

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return InterpolatedPredictionStrategy(train_inputs, train_prior_dist, train_labels, likelihood)
