from __future__ import annotations

import bisect
from typing import Callable, Tuple

import mujoco
import numpy as np
import torch
from torch import tensor

import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink


# predictive sampling
# https://arxiv.org/abs/2212.00541


class Policy:
    """Policy class for predictive sampling."""

    def __init__(
        self,
        naction: int,
        horizon: float,
        splinestep: float,
        interp: str = "zero",
        limits: np.ndarray | None = None):
        """Initialize policy class.

        Args:
          naction: number of actions
          horizon: planning horizon (seconds)
          splinestep: interval length between spline points
          interp (optional): type of action interpolation. Defaults to "zero".
          limits (optional): lower and upper bounds on actions. Defaults to None.
        """
        self._naction = naction
        self._splinestep = splinestep
        self._horizon = horizon
        self._nspline = int(horizon / splinestep) + 1
        self._parameters = np.zeros((self._naction, self._nspline), dtype=float)
        self._times = np.array(
          [t * self._splinestep for t in range(self._nspline)], dtype=float
        )
        self._interp = interp
        self._limits = limits

    @property
    def limits(self):
      return self._limits

    def _find_interval(
        self, sequence: np.ndarray, value: float
    ) -> Tuple[int, int]:
        """Find neighboring indices in sequence containing value.

        Args:
            sequence: array of values
            value: value to find in interval

        Returns:
            lower and upper indices in sequence containing value
        """
        # bisection search to get interval
        upper = bisect.bisect_right(sequence, value)
        lower = upper - 1

        # length of sequence
        L = len(sequence)

        # return feasible interval
        if lower < 0:
            return (0, 0)
        if lower > L - 1:
            return (L - 1, L - 1)
        return (max(lower, 0), min(upper, L - 1))

    def _slope(
        self, times: np.ndarray, params: np.ndarray, value: float
    ) -> np.ndarray:
        """Compute interpolated slope vector at value.

        Args:
            times: sequence of time markers
            params: sequence of vectors
            value: input where to compute slope

        Returns:
            interpolated slope vector
        """
        # bounds
        bounds = self._find_interval(times, value)

        times_length = len(times)

        # lower out of bounds
        if bounds[0] == 0 and bounds[1] == 0:
            if times_length > 2:
                return (params[:, bounds[1] + 1] - params[:, bounds[1]]) / (
                    times[bounds[1] + 1] - times[bounds[1]]
                )
            return np.zeros(params.shape[0])

        # upper out of bounds
        if bounds[0] == times_length - 1 and bounds[1] == times_length - 1:
            if times_length > 2:
                return (params[:, bounds[0]] - params[:, bounds[0] - 1]) / (
                    times[bounds[0]] - times[bounds[0] - 1]
                )
            return np.zeros(params.shape[0])

        # lower boundary
        if bounds[0] == 0:
            return (params[:, bounds[1]] - params[:, bounds[0]]) / (
                times[bounds[1]] - times[bounds[0]]
            )

        # internal interval
        return 0.5 * (params[:, bounds[1]] - params[:, bounds[0]]) / (
            times[bounds[1]] - times[bounds[0]]
        ) + 0.5 * (params[:, bounds[0]] - params[:, bounds[0] - 1]) / (
            times[bounds[0]] - times[bounds[0] - 1]
        )

    def ctrl(self, time: float) -> np.ndarray:
        return self.clamp(self._ctrl_from_action(self.action(time)))

    def _ctrl_from_action(self, action: np.ndarray) -> np.ndarray:
        max = self.limits[:, 0]
        min = self.limits[:, 1]
        # linear map
        return min + 0.5 * (max - min) * (action + 1)

        # quadratic map (p > 1, More weight to lower values)
        #k = 2
        #return min + (max - min) * ((action + 1) / 2) ** k

        # exponential map (More sensitivity near ±1)
        #k = 2
        #return min + (max - min) * (np.exp(k * action) - np.exp(-k)) / (np.exp(k) - np.exp(-k))

        # sigmoid map (smooth, changes slow near zero, faster near ±1.)
        #k = 2
        #return min + (max - min) * (1 / (1 + np.exp(-k * action)))

    def action(self, time: float) -> np.ndarray:
        """Return action from policy at time.

        Args:
            time: time value to evaluate plan for action

        Returns:
            interpolated action at time
        """
        # find interval containing time
        bounds = self._find_interval(self._times, time)

        # boundary case
        if bounds[0] == bounds[1]:
            return self._parameters[:, bounds[0]]

        # normalized time
        t = (time - self._times[bounds[0]]) / (
            self._times[bounds[1]] - self._times[bounds[0]]
        )

        if self._interp == "cubic":
            # spline coefficients
            c0 = 2.0 * t * t * t - 3.0 * t * t + 1.0
            c1 = (t * t * t - 2.0 * t * t + t) * (
                self._times[bounds[1]] - self._times[bounds[0]]
            )
            c2 = -2.0 * t * t * t + 3 * t * t
            c3 = (t * t * t - t * t) * (
                self._times[bounds[1]] - self._times[bounds[0]]
            )

            # slopes
            m0 = self._slope(self._times, self._parameters, self._times[bounds[0]])
            m1 = self._slope(self._times, self._parameters, self._times[bounds[1]])

            # interpolation
            return (c0 * self._parameters[:, bounds[0]]
                + c1 * m0
                + c2 * self._parameters[:, bounds[1]]
                + c3 * m1)
        elif self._interp == "linear":
            return ((1.0 - t) * self._parameters[:, bounds[0]]
                    + t * self._parameters[:, bounds[1]])
        else:  # self._interp == "zero"
            return self._parameters[:, bounds[0]]

    def resample(self, time: float):
        """Resample plan starting from time.

        Args:
          time: time value to start updated plan
        """
        # new times and parameters
        times = np.array(
          [i * self._splinestep + time for i in range(self._nspline)], dtype=float
        )
        parameters = np.vstack([self.action(t) for t in times]).T

        # update
        self._times = times
        self._parameters = parameters

    def add_noise(self, scale: float):
        """Add zero-mean Gaussian noise to plan.

        Args:
          scale: standard deviation of zero-mean Gaussian noise
        """
        # clamp within limits
        self._parameters = self._parameters \
            + np.random.normal(scale=scale, size=(self._naction, self._nspline))

    def noisy_copy(self, scale: float) -> Policy:
        """Return a copy of plan with added noise.

        Args:
          scale: standard deviation of zero-mean Gaussian noise

        Returns:
          copy of policy object with noisy plan
        """
        # create new policy object
        policy = Policy(self._naction, self._horizon, self._splinestep, self._interp, self._limits)

        # copy policy parameters into new object
        policy._parameters = np.copy(self._parameters)

        # get noisy parameters
        policy.add_noise(scale)

        return policy

    def clamp(self, action: np.ndarray) -> np.ndarray:
        """Return input clamped between limits.

        Args:
          action: input vector

        Returns:
          clamped input vector
        """
        # clamp within limits
        if self._limits is not None:
          return np.minimum(
              np.maximum(self._limits[:, 0], action), self._limits[:, 1]
          )
        return action

def rollout(
    qpos: np.ndarray,
    qvel: np.ndarray,
    time: float,
    timestep: float,
    scene: gs.Scene,
    entities: list[RigidEntity],
    model: mujoco.MjModel,
    reward: Callable,
    reset: Callable,
    apply_ctrl: Callable,
    init_state: Callable,
    end_state: Callable,
    policy: Policy,
    horizon: float,
) -> float:
    """Return total return by rollout out plan with forward dynamics.

    Args:
        qpos: initial configuration
        qvel: initial velocity
        time: current time
        timestep: current timestep
        scene: Genesis scene
        entities: list of entities involved
        model: MuJoCo model
        reward: function returning per-timestep reward value
        reset: function doing a custom reset of environment
        apply_ctrl: function applying ctrl to the system
        init_state: function initializing the system state at the beginning of a rollout
        end_state: function terminating the system state at the end of a rollout
        policy: plan for computing action at given time
        horizon: planning duration (seconds)

    Returns:
        total return (normalized by number of planning steps)
    """
    # number of steps
    steps = int(horizon / timestep)

    # initialize state
    init_state(entities, qpos, qvel)

    # initialize reward
    total_reward = 0.0

    # simulate
    for _ in range(steps):
        # reset
        reset(model, entities)

        # get ctrl from policy
        apply_ctrl(entities, policy.ctrl(time))

        # evaluate current reward
        total_reward += reward(model, entities)

        # step dynamics
        scene.step()

    # terminal reward
    end_state(entities)
    total_reward += reward(model, entities)

    return total_reward / (steps + 1)


class Planner:
    """Predictive sampling controller class."""

    def __init__(
        self,
        scene: gs.Scene,
        entities: list[RigidEntity | RigidLink],
        model: mujoco.MjModel,
        reward: Callable,
        reset: Callable,
        apply_ctrl: Callable,
        init_state: Callable,
        end_state: Callable,
        ndofs: int,
        dof_limits: np.ndarray,
        horizon: float,
        splinestep: float,
        planstep: float,
        nsample: int,
        noise_scale: float,
        nimprove: int,
        interp: str = "zero",
        limits: bool = True,
    ):
        """Initialize controller.

        Args:
            model: MuJoCo model
            reward: function returning per-timestep reward value
            reset: function customly resetting env
            apply_ctrl: function applying ctrl
            horizon: planning duration (seconds)
            splinestep: interval length between spline points
            planstep: interval length between forward dynamics steps
            nsample: number of noisy plans to evaluate
            noise_scale: standard deviation of zero-mean Gaussian
            nimprove: number of iterations to improve plan for fixed initial
            state
            interp: type of action interpolation. Defaults to
            "zero".
            limits: lower and upper bounds on action. Defaults to
            True.
        """
        self._scene = scene
        self._entities = entities
        self._model = model.__copy__()
        self._model.opt.timestep = planstep
        self._reward = reward
        self._reset = reset
        self._apply_ctrl = apply_ctrl
        self._init_state = init_state
        self._end_state = end_state
        self._ndofs = ndofs
        self._dof_limits = dof_limits
        self._horizon = horizon
        self.policy = Policy(
            ndofs,
            self._horizon,
            splinestep,
            interp=interp,
            limits=dof_limits,
        )
        self._nsample = nsample
        self._noise_scale = noise_scale
        self._nimprove = nimprove

    def action_from_policy(self, time: float) -> np.ndarray:
        """Return action at time from policy.

        Args:
          time: time to evaluate plan for action

        Returns:
          action interpolation at time
        """
        return self.policy.action(time)

    def ctrl_from_policy(self, time: float) -> np.ndarray:
        return self.policy.ctrl(time)

    def improve_policy(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        time: float,
        timestep: float
    ):
        """Iteratively improve plan via searching noisy plans.

        Args:
            qpos: initial configuration
            qvel: initial velocity
            time: current time
        """
        # resample
        self.policy.resample(time)

        for _ in range(self._nimprove):
            # evaluate nominal policy
            reward_nominal = rollout(
                qpos,
                qvel,
                time,
                timestep,
                self._scene,
                self._entities,
                self._model,
                self._reward,
                self._reset,
                self._apply_ctrl,
                self._init_state,
                self._end_state,
                self.policy,
                self._horizon
            )

        # evaluate noisy policies
        policies = []
        rewards = []
        for _ in range(self._nsample):
            # noisy policy
            noisy_policy = self.policy.noisy_copy(self._noise_scale)
            noisy_reward = rollout(
                qpos,
                qvel,
                time,
                timestep,
                self._scene,
                self._entities,
                self._model,
                self._reward,
                self._reset,
                self._apply_ctrl,
                self._init_state,
                self._end_state,
                noisy_policy,
                self._horizon
            )

            # collect result
            policies.append(noisy_policy)
            rewards.append(noisy_reward)

        # find best policy
        idx = np.argmax(rewards)

        # return new policy
        if rewards[idx] > reward_nominal:
            self.policy = policies[idx]
