"""Check muscle loading and recording with real PyElastica rods."""

from collections import defaultdict
from inspect import Parameter, signature

import numpy as np
import pytest
from elastica import (
    BaseSystemCollection,
    Constraints,
    CosseratRod,
    Forcing,
    OneEndFixedBC,
    PositionVerlet,
)

from coomm.actuations.muscles import (
    ApplyMuscles,
    ApplyMuscleGroups,
    MuscleForce,
    MuscleGroup,
)


def make_rod():
    # Only 0.2 requires nu; newer versions retain it as a rejected optional argument.
    nu = signature(CosseratRod.straight_rod).parameters.get("nu")
    kwargs = {"nu": 0.0} if nu is not None and nu.default is Parameter.empty else {}
    return CosseratRod.straight_rod(
        n_elements=8,
        start=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        base_length=0.2,
        base_radius=0.01,
        density=1000.0,
        youngs_modulus=1.0e6,
        shear_modulus=1.0e6 / 3.0,
        **kwargs,
    )


def make_muscle(rod):
    muscle = MuscleForce(
        ratio_muscle_position=np.tile([[0.5], [0.0], [0.0]], (1, rod.n_elems)),
        rest_muscle_area=np.full(rod.n_elems, 1.0e-6),
        max_muscle_stress=1.0e4,
    )
    muscle.set_current_length_as_rest_length(rod)
    muscle.apply_activation(0.4)
    return muscle


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("recording", ["omitted", "disabled", "enabled"])
def test_optional_callbacks_preserve_muscle_loads(grouped, recording):
    rod = make_rod()
    muscle = make_muscle(rod)
    actuations = [MuscleGroup(muscles=[muscle])] if grouped else [muscle]
    if grouped:
        actuations[0].apply_activation(0.4)
    callbacks = [defaultdict(list)]
    kwargs = {}
    if recording != "omitted":
        kwargs["callback_params_list"] = callbacks if recording == "enabled" else None
    forcing_class = ApplyMuscleGroups if grouped else ApplyMuscles
    forcing = forcing_class(actuations, step_skip=1, **kwargs)
    forcing.apply_torques(rod)

    np.testing.assert_allclose(muscle.muscle_force, 0.004)
    assert np.isfinite(rod.external_forces).all()
    assert np.isfinite(rod.external_torques).all()
    assert np.linalg.norm(rod.external_forces) > 0
    assert np.linalg.norm(rod.external_torques) > 0
    if recording == "enabled":
        np.testing.assert_array_equal(
            callbacks[0]["internal_force"][0], actuations[0].internal_force
        )
        if grouped:
            np.testing.assert_array_equal(
                callbacks[0]["muscles"][0]["internal_force"][0], muscle.internal_force
            )
    else:
        assert forcing.callback_params_list is None
        assert not callbacks[0]


def test_default_muscle_force_retains_area_scaling():
    rod = make_rod()
    muscle = make_muscle(rod)
    rod.dilatation[:] = 1.1
    muscle(rod)
    np.testing.assert_allclose(muscle.muscle_area, muscle.rest_muscle_area / 1.1)
    np.testing.assert_allclose(muscle.muscle_force, 0.004 / 1.1)


@pytest.mark.parametrize("recording", [False, True])
def test_muscle_group_short_rollout(recording):
    class Simulator(BaseSystemCollection, Constraints, Forcing):
        pass

    rod = make_rod()
    muscle = make_muscle(rod)
    group = MuscleGroup(muscles=[muscle])
    group.apply_activation(0.4)
    callbacks = [defaultdict(list)]
    simulator = Simulator()
    simulator.append(rod)
    simulator.constrain(rod).using(
        OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )
    simulator.add_forcing_to(rod).using(
        ApplyMuscleGroups,
        muscle_groups=[group],
        step_skip=1,
        callback_params_list=callbacks if recording else None,
    )
    simulator.finalize()
    initial_position = rod.position_collection.copy()
    stepper = PositionVerlet()
    advance = getattr(stepper, "step", None)
    if advance is None:
        # PyElastica 0.2 uses the original stepping interface.
        from elastica.timestepper import extend_stepper_interface

        do_step, stages = extend_stepper_interface(stepper, simulator)

        def advance(system, time, dt):
            return do_step(stepper, stages, system, time, dt)

    time = 0.0
    for _ in range(50):
        time = advance(simulator, time, 1.0e-4)
    assert time == pytest.approx(0.005)
    assert np.isfinite(rod.position_collection).all()
    assert np.isfinite(rod.velocity_collection).all()
    assert np.linalg.norm(rod.position_collection - initial_position) > 1.0e-10
    if recording:
        assert callbacks[0]["internal_force"]
        assert callbacks[0]["muscles"][0]["muscle_length"]
    else:
        assert not callbacks[0]
