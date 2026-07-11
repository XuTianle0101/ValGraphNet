"""Analytic tetrahedral mechanics used by the CHP-GNS physical decoder.

The routines in this module use a total-Lagrangian convention.  A tetrahedron
stores ``Dm = [X1-X0, X2-X0, X3-X0]`` in the reference configuration and the
deformation gradient is ``F = Ds @ inv(Dm)``.  Energy densities and first
Piola stresses are per unit reference volume.

All determinant, inverse, and stress calculations are promoted to FP32 when
their inputs are FP16/BF16.  FP64 inputs are retained for verification and
finite-difference tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as functional


__all__ = [
    "AnalyticPotential",
    "ConstitutiveResponse",
    "DeformationInvariants",
    "IntegrationResult",
    "TetrahedralReference",
    "assemble_internal_force",
    "deformation_gradient",
    "invariants",
    "negative_j_barrier",
    "precompute_tetrahedra",
    "project_cell_to_nodes",
    "semi_implicit_step",
    "von_mises",
]


@dataclass(frozen=True)
class TetrahedralReference:
    """Reference quantities that are constant throughout a rollout."""

    dm_inv: torch.Tensor
    volume: torch.Tensor
    shape_gradients: torch.Tensor
    lumped_mass: torch.Tensor


@dataclass(frozen=True)
class DeformationInvariants:
    """Raw and isochoric right-Cauchy--Green invariants."""

    c: torch.Tensor
    i1: torch.Tensor
    i2: torch.Tensor
    j: torch.Tensor
    i1_bar: torch.Tensor
    i2_bar: torch.Tensor


@dataclass(frozen=True)
class ConstitutiveResponse:
    """Energy and stresses produced by one evaluation of a potential."""

    energy_density: torch.Tensor
    first_piola: torch.Tensor
    cauchy_stress: torch.Tensor
    invariants: DeformationInvariants
    inversion_barrier: torch.Tensor

    @property
    def energy(self) -> torch.Tensor:
        """Alias retained for concise downstream loss code."""

        return self.energy_density

    @property
    def stress(self) -> torch.Tensor:
        """Alias for the full Cauchy stress tensor."""

        return self.cauchy_stress

    @property
    def p(self) -> torch.Tensor:
        """Alias for the first Piola--Kirchhoff stress."""

        return self.first_piola


@dataclass(frozen=True)
class IntegrationResult:
    """State returned by :func:`semi_implicit_step`."""

    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.position
        yield self.velocity
        yield self.acceleration


def _work_dtype(tensor: torch.Tensor) -> torch.dtype:
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if tensor.dtype not in (torch.float32, torch.float64):
        return torch.float32
    return tensor.dtype


def _as_float_tensor(value, *, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=like.device, dtype=_work_dtype(like))


def precompute_tetrahedra(
    nodes: torch.Tensor,
    cells: torch.Tensor,
    density: float | torch.Tensor = 1.0,
) -> TetrahedralReference:
    """Precompute reference geometry and a diagonal lumped mass.

    Args:
        nodes: Reference coordinates with shape ``[N, 3]``.
        cells: Four-node connectivity with shape ``[M, 4]``.
        density: Scalar or one density value per tetrahedron.

    Returns:
        Reference inverse matrices, positive cell volumes, gradients of all
        four linear shape functions, and one lumped mass per node.
    """

    nodes = torch.as_tensor(nodes)
    cells = torch.as_tensor(cells, device=nodes.device, dtype=torch.long)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must have shape [N, 3]")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("cells must have shape [M, 4]")
    if cells.numel() == 0:
        raise ValueError("at least one tetrahedral cell is required")
    if int(cells.min()) < 0 or int(cells.max()) >= nodes.shape[0]:
        raise IndexError("cells contain a node index outside nodes")

    work_nodes = nodes.to(dtype=_work_dtype(nodes))
    vertices = work_nodes[cells]
    dm = torch.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ),
        dim=-1,
    )
    determinant = torch.linalg.det(dm)
    tolerance = 100.0 * torch.finfo(work_nodes.dtype).eps
    if bool(torch.any(determinant.abs() <= tolerance)):
        raise ValueError("cells contain a degenerate tetrahedron")

    dm_inv = torch.linalg.inv(dm)
    volume = determinant.abs() / 6.0

    # grad(N1), grad(N2), and grad(N3) are the rows of inv(Dm).
    gradients_123 = dm_inv
    gradient_0 = -gradients_123.sum(dim=1, keepdim=True)
    shape_gradients = torch.cat((gradient_0, gradients_123), dim=1)

    cell_density = _as_float_tensor(density, like=work_nodes)
    if cell_density.ndim == 0:
        cell_density = cell_density.expand(cells.shape[0])
    elif cell_density.shape != (cells.shape[0],):
        raise ValueError("density must be scalar or have shape [M]")
    if bool(torch.any(cell_density <= 0.0)):
        raise ValueError("density must be strictly positive")

    nodal_contribution = (cell_density * volume / 4.0)[:, None].expand(-1, 4)
    lumped_mass = work_nodes.new_zeros(work_nodes.shape[0])
    lumped_mass.index_add_(0, cells.reshape(-1), nodal_contribution.reshape(-1))

    return TetrahedralReference(
        dm_inv=dm_inv,
        volume=volume,
        shape_gradients=shape_gradients,
        lumped_mass=lumped_mass,
    )


def deformation_gradient(
    current_pos: torch.Tensor,
    cells: torch.Tensor,
    dm_inv: torch.Tensor,
) -> torch.Tensor:
    """Compute ``F = Ds @ Dm_inv`` for one state or a leading batch."""

    current_pos = torch.as_tensor(current_pos)
    if current_pos.ndim < 2 or current_pos.shape[-1] != 3:
        raise ValueError("current_pos must have shape [..., N, 3]")
    cells = torch.as_tensor(cells, device=current_pos.device, dtype=torch.long)
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("cells must have shape [M, 4]")
    work_position = current_pos.to(dtype=_work_dtype(current_pos))
    work_dm_inv = torch.as_tensor(
        dm_inv, device=current_pos.device, dtype=work_position.dtype
    )
    if work_dm_inv.shape != (cells.shape[0], 3, 3):
        raise ValueError("dm_inv must have shape [M, 3, 3]")

    vertices = work_position[..., cells, :]
    ds = torch.stack(
        (
            vertices[..., 1, :] - vertices[..., 0, :],
            vertices[..., 2, :] - vertices[..., 0, :],
            vertices[..., 3, :] - vertices[..., 0, :],
        ),
        dim=-1,
    )
    return ds @ work_dm_inv


def invariants(deformation: torch.Tensor, eps: float | None = None) -> DeformationInvariants:
    """Return objective invariants of a deformation gradient.

    ``i1_bar`` and ``i2_bar`` use a clamped positive determinant so invalid
    elements remain finite and can be driven back by an inversion barrier.
    The unclamped signed determinant is always returned as ``j``.
    """

    deformation = torch.as_tensor(deformation)
    if deformation.shape[-2:] != (3, 3):
        raise ValueError("deformation must have shape [..., 3, 3]")
    work_f = deformation.to(dtype=_work_dtype(deformation))
    c = work_f.transpose(-1, -2) @ work_f
    i1 = torch.diagonal(c, dim1=-2, dim2=-1).sum(dim=-1)
    trace_c2 = (c * c.transpose(-1, -2)).sum(dim=(-2, -1))
    i2 = 0.5 * (i1.square() - trace_c2)
    j = torch.linalg.det(work_f)
    minimum = float(eps) if eps is not None else 10.0 * torch.finfo(work_f.dtype).eps
    j_safe = j.clamp_min(minimum)
    return DeformationInvariants(
        c=c,
        i1=i1,
        i2=i2,
        j=j,
        i1_bar=i1 * j_safe.pow(-2.0 / 3.0),
        i2_bar=i2 * j_safe.pow(-4.0 / 3.0),
    )


def _cofactor(deformation: torch.Tensor) -> torch.Tensor:
    """Derivative of determinant with respect to the input matrix."""

    column_0 = deformation[..., :, 0]
    column_1 = deformation[..., :, 1]
    column_2 = deformation[..., :, 2]
    return torch.stack(
        (
            torch.linalg.cross(column_1, column_2, dim=-1),
            torch.linalg.cross(column_2, column_0, dim=-1),
            torch.linalg.cross(column_0, column_1, dim=-1),
        ),
        dim=-1,
    )


def negative_j_barrier(
    j: torch.Tensor,
    minimum_j: float = 0.0,
    stiffness: float | torch.Tensor = 1.0,
    reduction: str = "none",
) -> torch.Tensor:
    """Quadratic penalty for determinants below ``minimum_j``.

    The unreduced result is an energy density
    ``0.5 * stiffness * relu(minimum_j - J)^2``.
    """

    j = torch.as_tensor(j)
    work_j = j.to(dtype=_work_dtype(j))
    penalty = 0.5 * torch.as_tensor(
        stiffness, dtype=work_j.dtype, device=work_j.device
    ) * functional.relu(float(minimum_j) - work_j).square()
    if reduction == "none":
        return penalty
    if reduction == "mean":
        return penalty.mean()
    if reduction == "sum":
        return penalty.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _initial_coefficients(
    value: float | Sequence[float], order: int, name: str
) -> torch.Tensor:
    if order < 1:
        raise ValueError(f"{name} order must be at least one")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        coefficients = torch.as_tensor(list(value), dtype=torch.float32)
        if coefficients.shape != (order,):
            raise ValueError(f"{name} must contain exactly {order} values")
    else:
        base = float(value)
        coefficients = torch.tensor(
            [base * (0.1**index) for index in range(order)], dtype=torch.float32
        )
    if bool(torch.any(coefficients <= 0.0)):
        raise ValueError(f"{name} coefficients must be strictly positive")
    return coefficients


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    # log(expm1(x)) loses precision for large x; this equivalent form is stable.
    return value + torch.log(-torch.expm1(-value))


def _polynomial_energy_and_derivative(
    value: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    exponent_stride: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    if exponent_stride < 1:
        raise ValueError("exponent_stride must be positive")
    energy = torch.zeros_like(value)
    derivative = torch.zeros_like(value)
    for index, coefficient in enumerate(coefficients):
        exponent = exponent_stride * (index + 1)
        energy = energy + coefficient * value.pow(exponent)
        derivative = derivative + coefficient * exponent * value.pow(exponent - 1)
    return energy, derivative


class AnalyticPotential(nn.Module):
    """Objective hyperelastic potential with an explicit first derivative.

    All learned coefficients are represented through ``softplus`` and are
    therefore positive.  The module computes first Piola and Cauchy stress by
    closed-form expressions; it never calls ``autograd.grad`` with respect to
    ``F``.  Autograd still propagates normally to the trainable coefficients.

    Fiber terms are optional and use the objective invariant
    ``I4 = ||F a0||^2``.  A direction can be registered at construction time
    or supplied per cell to :meth:`forward`.
    """

    def __init__(
        self,
        order: int = 2,
        i1_init: float | Sequence[float] = 0.5,
        i2_init: float | Sequence[float] = 0.1,
        j_init: float | Sequence[float] = 1.0,
        log_j_init: float = 1.0,
        *,
        fiber_order: int = 0,
        fiber_init: float | Sequence[float] = 0.1,
        fiber_direction: torch.Tensor | None = None,
        inversion_stiffness: float = 10.0,
        minimum_j: float = 0.0,
        determinant_eps: float | None = None,
        minimum_coefficient: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.order = int(order)
        self.fiber_order = int(fiber_order)
        self.inversion_stiffness = float(inversion_stiffness)
        self.minimum_j = float(minimum_j)
        self.determinant_eps = determinant_eps
        self.minimum_coefficient = float(minimum_coefficient)
        if self.inversion_stiffness < 0.0:
            raise ValueError("inversion_stiffness must be non-negative")
        if self.minimum_coefficient < 0.0:
            raise ValueError("minimum_coefficient must be non-negative")
        if self.fiber_order < 0:
            raise ValueError("fiber_order must be non-negative")

        self.raw_i1 = nn.Parameter(
            _inverse_softplus(_initial_coefficients(i1_init, self.order, "i1_init"))
        )
        self.raw_i2 = nn.Parameter(
            _inverse_softplus(_initial_coefficients(i2_init, self.order, "i2_init"))
        )
        self.raw_j = nn.Parameter(
            _inverse_softplus(_initial_coefficients(j_init, self.order, "j_init"))
        )
        log_j = torch.tensor(float(log_j_init), dtype=torch.float32)
        if float(log_j) <= 0.0:
            raise ValueError("log_j_init must be strictly positive")
        self.raw_log_j = nn.Parameter(_inverse_softplus(log_j))

        if self.fiber_order:
            self.raw_fiber = nn.Parameter(
                _inverse_softplus(
                    _initial_coefficients(fiber_init, self.fiber_order, "fiber_init")
                )
            )
        else:
            self.register_parameter("raw_fiber", None)

        if fiber_direction is None:
            self.register_buffer("default_fiber_direction", None)
        else:
            direction = torch.as_tensor(fiber_direction, dtype=torch.float32)
            if direction.shape[-1:] != (3,):
                raise ValueError("fiber_direction must have shape [..., 3]")
            self.register_buffer("default_fiber_direction", direction)

    @property
    def i1_coefficients(self) -> torch.Tensor:
        return functional.softplus(self.raw_i1) + self.minimum_coefficient

    @property
    def i2_coefficients(self) -> torch.Tensor:
        return functional.softplus(self.raw_i2) + self.minimum_coefficient

    @property
    def j_coefficients(self) -> torch.Tensor:
        return functional.softplus(self.raw_j) + self.minimum_coefficient

    @property
    def log_j_coefficient(self) -> torch.Tensor:
        return functional.softplus(self.raw_log_j) + self.minimum_coefficient

    @property
    def fiber_coefficients(self) -> torch.Tensor:
        if self.raw_fiber is None:
            return self.raw_i1.new_empty(0)
        return functional.softplus(self.raw_fiber) + self.minimum_coefficient

    def forward(
        self,
        deformation: torch.Tensor,
        fiber_direction: torch.Tensor | None = None,
    ) -> ConstitutiveResponse:
        deformation = torch.as_tensor(deformation)
        if deformation.shape[-2:] != (3, 3):
            raise ValueError("deformation must have shape [..., 3, 3]")
        work_f = deformation.to(dtype=_work_dtype(deformation))
        state = invariants(work_f, eps=self.determinant_eps)
        eps = (
            float(self.determinant_eps)
            if self.determinant_eps is not None
            else 10.0 * torch.finfo(work_f.dtype).eps
        )
        j_safe = state.j.clamp_min(eps)
        valid_j = (state.j > eps).to(work_f.dtype)
        cofactor = _cofactor(work_f)
        d_j_safe = cofactor * valid_j[..., None, None]

        x1 = state.i1_bar - 3.0
        x2 = state.i2_bar - 3.0
        xj = state.j - 1.0
        energy_i1, derivative_i1 = _polynomial_energy_and_derivative(
            x1, self.i1_coefficients, exponent_stride=1
        )
        energy_i2, derivative_i2 = _polynomial_energy_and_derivative(
            x2, self.i2_coefficients, exponent_stride=1
        )
        energy_j, derivative_j = _polynomial_energy_and_derivative(
            xj, self.j_coefficients
        )

        identity = torch.eye(3, dtype=work_f.dtype, device=work_f.device)
        derivative_i1_bar = (
            2.0 * j_safe.pow(-2.0 / 3.0)[..., None, None] * work_f
            - (2.0 / 3.0)
            * state.i1[..., None, None]
            * j_safe.pow(-5.0 / 3.0)[..., None, None]
            * d_j_safe
        )
        derivative_i2_bar = (
            2.0
            * j_safe.pow(-4.0 / 3.0)[..., None, None]
            * (work_f @ (state.i1[..., None, None] * identity - state.c))
            - (4.0 / 3.0)
            * state.i2[..., None, None]
            * j_safe.pow(-7.0 / 3.0)[..., None, None]
            * d_j_safe
        )

        log_energy = self.log_j_coefficient * (-torch.log(j_safe) + j_safe - 1.0)
        log_derivative = (
            self.log_j_coefficient
            * (-j_safe.reciprocal() + 1.0)
            * valid_j
        )
        first_piola = (
            derivative_i1[..., None, None] * derivative_i1_bar
            + derivative_i2[..., None, None] * derivative_i2_bar
            + (derivative_j + log_derivative)[..., None, None] * cofactor
        )
        energy_density = energy_i1 + energy_i2 + energy_j + log_energy

        direction = fiber_direction
        if direction is None:
            direction = self.default_fiber_direction
        if self.fiber_order:
            if direction is None:
                raise ValueError("fiber_direction is required when fiber_order is non-zero")
            direction = torch.as_tensor(
                direction, device=work_f.device, dtype=work_f.dtype
            )
            if direction.shape[-1:] != (3,):
                raise ValueError("fiber_direction must have shape [..., 3]")
            norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
            if bool(torch.any(norm <= torch.finfo(work_f.dtype).eps)):
                raise ValueError("fiber directions must be non-zero")
            direction = direction / norm
            deformed_fiber = (work_f @ direction[..., None]).squeeze(-1)
            i4_minus_one = deformed_fiber.square().sum(dim=-1) - 1.0
            fiber_energy, fiber_derivative = _polynomial_energy_and_derivative(
                i4_minus_one, self.fiber_coefficients
            )
            derivative_i4 = 2.0 * (
                deformed_fiber[..., :, None] * direction[..., None, :]
            )
            energy_density = energy_density + fiber_energy
            first_piola = first_piola + fiber_derivative[..., None, None] * derivative_i4

        inversion_barrier = negative_j_barrier(
            state.j,
            minimum_j=self.minimum_j,
            stiffness=self.inversion_stiffness,
            reduction="none",
        )
        barrier_violation = functional.relu(self.minimum_j - state.j)
        barrier_derivative = -self.inversion_stiffness * barrier_violation
        energy_density = energy_density + inversion_barrier
        first_piola = first_piola + barrier_derivative[..., None, None] * cofactor

        cauchy_stress = (first_piola @ work_f.transpose(-1, -2)) / j_safe[
            ..., None, None
        ]
        cauchy_stress = 0.5 * (cauchy_stress + cauchy_stress.transpose(-1, -2))
        return ConstitutiveResponse(
            energy_density=energy_density,
            first_piola=first_piola,
            cauchy_stress=cauchy_stress,
            invariants=state,
            inversion_barrier=inversion_barrier,
        )


def assemble_internal_force(
    first_piola: torch.Tensor,
    cells: torch.Tensor,
    volume: torch.Tensor,
    shape_gradients: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Assemble conservative nodal force ``-sum(V P grad(N))``.

    ``first_piola`` may have leading batch dimensions.  Reference geometry is
    shared across those dimensions.
    """

    first_piola = torch.as_tensor(first_piola)
    if first_piola.shape[-2:] != (3, 3):
        raise ValueError("first_piola must have shape [..., M, 3, 3]")
    cells = torch.as_tensor(cells, device=first_piola.device, dtype=torch.long)
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("cells must have shape [M, 4]")
    if first_piola.shape[-3] != cells.shape[0]:
        raise ValueError("first_piola cell dimension does not match cells")
    if num_nodes < 1 or (cells.numel() and int(cells.max()) >= num_nodes):
        raise ValueError("num_nodes is inconsistent with cells")

    work_p = first_piola.to(dtype=_work_dtype(first_piola))
    work_volume = torch.as_tensor(
        volume, device=work_p.device, dtype=work_p.dtype
    )
    work_gradients = torch.as_tensor(
        shape_gradients, device=work_p.device, dtype=work_p.dtype
    )
    if work_volume.shape != (cells.shape[0],):
        raise ValueError("volume must have shape [M]")
    if work_gradients.shape != (cells.shape[0], 4, 3):
        raise ValueError("shape_gradients must have shape [M, 4, 3]")

    contribution = -torch.einsum("...mij,mnj->...mni", work_p, work_gradients)
    contribution = contribution * work_volume[..., None, None]
    leading_shape = contribution.shape[:-3]
    batch_size = 1
    for size in leading_shape:
        batch_size *= size
    flat_contribution = contribution.reshape(batch_size, -1, 3)
    batch_offset = torch.arange(batch_size, device=work_p.device)[:, None] * num_nodes
    indices = cells.reshape(1, -1) + batch_offset
    assembled = work_p.new_zeros(batch_size * num_nodes, 3)
    assembled.index_add_(
        0,
        indices.reshape(-1),
        flat_contribution.reshape(-1, 3),
    )
    result = assembled.reshape(*leading_shape, num_nodes, 3)
    return result


def project_cell_to_nodes(
    cell_values: torch.Tensor,
    cells: torch.Tensor,
    num_nodes: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project cell values to nodes by a (possibly weighted) average.

    The first dimension of ``cell_values`` is the cell dimension.  Arbitrary
    trailing feature dimensions, including full ``3x3`` stresses, are kept.
    """

    cell_values = torch.as_tensor(cell_values)
    cells = torch.as_tensor(cells, device=cell_values.device, dtype=torch.long)
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("cells must have shape [M, 4]")
    if cell_values.ndim < 1 or cell_values.shape[0] != cells.shape[0]:
        raise ValueError("cell_values must have shape [M, ...]")
    if num_nodes < 1 or (cells.numel() and int(cells.max()) >= num_nodes):
        raise ValueError("num_nodes is inconsistent with cells")
    if weights is None:
        weights = cell_values.new_ones(cells.shape[0])
    else:
        weights = torch.as_tensor(
            weights, device=cell_values.device, dtype=cell_values.dtype
        )
        if weights.shape != (cells.shape[0],):
            raise ValueError("weights must have shape [M]")
        if bool(torch.any(weights < 0.0)):
            raise ValueError("weights must be non-negative")

    feature_shape = cell_values.shape[1:]
    expanded_weights = weights.reshape(-1, *((1,) * len(feature_shape)))
    contribution = (cell_values * expanded_weights)[:, None, ...].expand(
        -1, 4, *feature_shape
    )
    projected = cell_values.new_zeros((num_nodes, *feature_shape))
    projected.index_add_(0, cells.reshape(-1), contribution.reshape(-1, *feature_shape))
    denominator = cell_values.new_zeros(num_nodes)
    denominator.index_add_(0, cells.reshape(-1), weights[:, None].expand(-1, 4).reshape(-1))
    denominator = denominator.clamp_min(torch.finfo(cell_values.dtype).eps)
    return projected / denominator.reshape(num_nodes, *((1,) * len(feature_shape)))


def von_mises(stress: torch.Tensor) -> torch.Tensor:
    """Compute von-Mises equivalent stress from full ``3x3`` tensors."""

    stress = torch.as_tensor(stress)
    if stress.shape[-2:] != (3, 3):
        raise ValueError("stress must have shape [..., 3, 3]")
    work_stress = stress.to(dtype=_work_dtype(stress))
    symmetric = 0.5 * (work_stress + work_stress.transpose(-1, -2))
    mean_stress = torch.diagonal(symmetric, dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    identity = torch.eye(3, dtype=symmetric.dtype, device=symmetric.device)
    deviator = symmetric - mean_stress[..., None, None] * identity
    equivalent_squared = (1.5 * deviator.square().sum(dim=(-2, -1))).clamp_min(0.0)
    # Plain sqrt has an infinite derivative at the physically important
    # zero-stress reference state.  Subtracting the identical smoothing offset
    # preserves an exact zero value while keeping training gradients finite.
    eps = equivalent_squared.new_tensor(torch.finfo(equivalent_squared.dtype).eps)
    return torch.sqrt(equivalent_squared + eps) - torch.sqrt(eps)


def semi_implicit_step(
    position: torch.Tensor,
    velocity: torch.Tensor,
    force: torch.Tensor,
    mass: torch.Tensor,
    dt: float | torch.Tensor,
    *,
    substeps: int = 1,
    fixed_mask: torch.Tensor | None = None,
    prescribed_mask: torch.Tensor | None = None,
    prescribed_position: torch.Tensor | None = None,
    prescribed_velocity: torch.Tensor | None = None,
) -> IntegrationResult:
    """Advance a nodal state with a constant force and symplectic Euler.

    Fixed nodes retain their input position and receive zero velocity.
    Prescribed values are applied last and therefore take precedence if masks
    overlap.  When only a prescribed position is given, its velocity is
    derived from the exact one-step displacement.
    """

    position = torch.as_tensor(position)
    velocity = torch.as_tensor(velocity, device=position.device)
    force = torch.as_tensor(force, device=position.device)
    if position.shape != velocity.shape or position.shape != force.shape:
        raise ValueError("position, velocity, and force must have the same shape")
    if position.ndim < 2 or position.shape[-1] != 3:
        raise ValueError("state tensors must have shape [..., N, 3]")
    if int(substeps) < 1:
        raise ValueError("substeps must be at least one")

    dtype = _work_dtype(position)
    initial_position = position.to(dtype=dtype)
    initial_velocity = velocity.to(dtype=dtype)
    work_force = force.to(dtype=dtype)
    work_mass = torch.as_tensor(mass, device=position.device, dtype=dtype)
    if work_mass.shape == initial_position.shape[:-1]:
        work_mass = work_mass[..., None]
    elif work_mass.shape != (*initial_position.shape[:-1], 1):
        raise ValueError("mass must have shape [..., N] or [..., N, 1]")
    if bool(torch.any(work_mass <= 0.0)):
        raise ValueError("mass must be strictly positive")
    step_dt = torch.as_tensor(dt, device=position.device, dtype=dtype)
    if step_dt.numel() != 1 or float(step_dt) <= 0.0:
        raise ValueError("dt must be a positive scalar")

    acceleration = work_force / work_mass
    next_position = initial_position
    next_velocity = initial_velocity
    substep_dt = step_dt / int(substeps)
    for _ in range(int(substeps)):
        next_velocity = next_velocity + substep_dt * acceleration
        next_position = next_position + substep_dt * next_velocity

    if fixed_mask is not None:
        fixed = torch.as_tensor(fixed_mask, device=position.device, dtype=torch.bool)
        if fixed.shape != initial_position.shape[:-1]:
            raise ValueError("fixed_mask must have shape [..., N]")
        fixed = fixed[..., None]
        next_position = torch.where(fixed, initial_position, next_position)
        next_velocity = torch.where(fixed, torch.zeros_like(next_velocity), next_velocity)

    if prescribed_mask is not None:
        prescribed = torch.as_tensor(
            prescribed_mask, device=position.device, dtype=torch.bool
        )
        if prescribed.shape != initial_position.shape[:-1]:
            raise ValueError("prescribed_mask must have shape [..., N]")
        if prescribed_position is None and prescribed_velocity is None:
            raise ValueError("a prescribed position or velocity is required")
        prescribed = prescribed[..., None]
        if prescribed_position is not None:
            target_position = torch.as_tensor(
                prescribed_position, device=position.device, dtype=dtype
            )
            if target_position.shape != initial_position.shape:
                raise ValueError("prescribed_position must match position")
            next_position = torch.where(prescribed, target_position, next_position)
        if prescribed_velocity is not None:
            target_velocity = torch.as_tensor(
                prescribed_velocity, device=position.device, dtype=dtype
            )
            if target_velocity.shape != initial_velocity.shape:
                raise ValueError("prescribed_velocity must match velocity")
        elif prescribed_position is not None:
            target_velocity = (target_position - initial_position) / step_dt
        else:
            target_velocity = next_velocity
        next_velocity = torch.where(prescribed, target_velocity, next_velocity)

    effective_acceleration = (next_velocity - initial_velocity) / step_dt
    return IntegrationResult(
        position=next_position,
        velocity=next_velocity,
        acceleration=effective_acceleration,
    )
