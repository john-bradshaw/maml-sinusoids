import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import numpy as np
import optax
from torch.utils import data
import tqdm


@dataclass
class Params:
    # model parameters
    nn_num_units: int = 40

    # inner loop parameters
    il_num_steps: int = 1
    il_lr: float = 0.01  # p. 5

    # outer loop parameters
    ol_lr = 3e-4
    num_epochs = 100
    num_tasks_per_batch = 5

    # task setup
    num_training_tasks = 100
    num_data_points_per_task: int = 5



def main(run_params: Params):
    rng = np.random.RandomState(76324)
    jxkey = random.PRNGKey(0)

    amplitude_limits = (0.1, 5.0)
    phase_limits = (0, np.pi)
    x_limits = (-5, 5)

    class SinusoidData(data.Dataset):
        def __init__(self, rng, amplitude, phase, x_limits, K=5):
            self.amplitude = amplitude
            self.phase = phase
            self.x_limits = x_limits

            self.x = rng.uniform(*x_limits, K)
            self.y = self.evaluate_anywhere(self.x)
            self._rng = rng

        def __getitem__(self, idx):
            x = self.x[idx]
            y = self.y[idx]
            return (x, y)

        def __len__(self):
            return len(self.x)

        def evaluate_anywhere(self, x_locations: np.array):
            return self.amplitude * np.sin(x_locations + self.phase)

        @classmethod
        def create_dataset_generator(cls, rng, amplitude_limits, phase_limits, x_limits, K=5):
            while True:
                amplitude = rng.uniform(*amplitude_limits)
                phase = rng.uniform(*phase_limits)
                yield cls(rng, amplitude, phase, x_limits, K)

        def create_equiv_dataset(self, **kwargs):
            _kwargs = dict(rng=self._rng, amplitude=self.amplitude, phase=self.phase,
                               x_limits=self.x_limits, K=len(self))
            _kwargs.update(kwargs)
            return self.__class__(**_kwargs)

        def split_off_test(self, test_size, remove_from_self=False):
            assert test_size <= len(self)
            test = self.create_equiv_dataset()
            test.x = test.x[:test_size]
            test.y = test.y[:test_size]
            if remove_from_self:
                self.x = self.x[test_size:]
                self.y = self.y[test_size:]
            return test


    data_generator = SinusoidData.create_dataset_generator(rng, amplitude_limits, phase_limits, x_limits,
                                                           K=run_params.num_data_points_per_task)

    test_examples = [next(data_generator) for _ in
                     range(3)]  # <-- we'll show the method working at the end on 5 examples
    grid = jnp.linspace(*x_limits, 100)

    class FFN(nn.Module):
        layer_sizes: typing.Sequence[int]

        @nn.compact
        def __call__(self, x):
            for layer_size in self.layer_sizes[:-1]:
                x = nn.Dense(layer_size)(x)
                x = nn.relu(x)
            x = nn.Dense(self.layer_sizes[-1])(x)  # <- no non-linearity on output.
            return x

    feed_forward_net = FFN([run_params.nn_num_units, 1])

    def create_train_state(learning_rate, optimizer='adam', params=None, jxkey=None):
        """Creates initial `TrainState`."""

        if params is None:
            assert jxkey is not None, "If creating new parameters need to be given random key."
            x = jnp.ones((1,))  # Dummy input
            params = feed_forward_net.init(jxkey, x)['params']

        if optimizer == 'adam':
            tx = optax.adam(learning_rate)
        elif optimizer == 'sgd':
            tx = optax.sgd(learning_rate)
        else:
            raise NotImplementedError
        return train_state.TrainState.create(
            apply_fn=feed_forward_net.apply, params=params, tx=tx)

    def make_mse_func(ts, x_batched, y_batched):
        # see: https://flax.readthedocs.io/en/latest/notebooks/flax_basics.html#Linear-regression-with-Flax
        def mse(net_params):
            preds = ts.apply_fn({"params": net_params}, x_batched)
            err = preds - y_batched
            return jnp.mean(err**2)
        return mse

    def optimize_on_batch(net_params, x_batched_context, y_batched_context, num_steps):
        ts_inner = create_train_state(run_params.il_lr, optimizer='sgd', params=net_params)

        context_loss = make_mse_func(ts_inner, x_batched_context, y_batched_context)
        inner_loop_grad_fn = jax.value_and_grad(context_loss)
        for i in range(num_steps):
            loss_val, grads = inner_loop_grad_fn(ts_inner.params)
            ts_inner = ts_inner.apply_gradients(grads=grads)
        return ts_inner.params


    def make_inner_loop_func(ts, x_batched_context, y_batched_context, x_batched_test, y_batched_test):
        def il_(net_params):
            net_params = optimize_on_batch(net_params, x_batched_context, y_batched_context, run_params.il_num_steps)
            test_loss = make_mse_func(ts, x_batched_test, y_batched_test)(net_params)
            return test_loss
        return il_

    def make_outer_loop_eval_loss(ts, tasks):
        def ol_(net_params):
            total_loss = 0.
            for task in tasks:
                test_task = task.split_off_test(run_params.num_data_points_per_task)

                data = []
                for t_ in [task, test_task]:
                    xs, ys = zip(*(t_[i] for i in range(run_params.num_data_points_per_task)))
                    xs = jnp.vstack(list(xs))
                    ys = jnp.vstack(list(ys))
                    data += [xs, ys]
                il = make_inner_loop_func(ts, *data)
                total_loss += il(net_params)
            return total_loss / len(tasks)
        return ol_

    def ol_train_step(ts, tasks_for_batch):
        """Train for a single step."""
        ol_step = make_outer_loop_eval_loss(ts, tasks_for_batch)
        grad_fn = jax.value_and_grad(ol_step)
        loss, grads = grad_fn(ts.params)
        ts = ts.apply_gradients(grads=grads)
        return ts, loss

    def train_epoch(ts, train_tasks, num_tasks_per_batch, epoch_id, rng):
        train_ds_size = len(train_tasks)
        steps_per_epoch = train_ds_size // num_tasks_per_batch

        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[:steps_per_epoch * num_tasks_per_batch]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, num_tasks_per_batch))
        losses = []
        for perm in tqdm.tqdm(perms, desc=f"epoch {epoch_id}"):
            tasks_for_batch = [train_tasks[i] for i in perm]
            ts, loss = ol_train_step(ts, tasks_for_batch)
            losses.append(loss)

        # compute mean of metrics across each batch in epoch.
        batch_metrics_np = jax.device_get(losses)
        mean_loss = np.mean(batch_metrics_np)

        print(f'train epoch: {epoch_id}, loss: {mean_loss})')
        return ts

    def inner_loop_one_task(ts, task, num_steps):
        xs, ys = zip(*(task[i] for i in range(len(task))))
        xs = jnp.vstack(list(xs))
        ys = jnp.vstack(list(ys))
        net_params = optimize_on_batch(ts.params, xs, ys, num_steps)
        ts = ts.replace(params=net_params)
        return ts

    def eval_model_on_grid(ts, grid, tasks):
        mses = []
        for task in tasks:
            new_ts = inner_loop_one_task(ts, task, 1)
            preds = new_ts.apply_fn({"params" : new_ts.params}, grid[:, None])
            true_values = task.evaluate_anywhere(np.array(grid))
            mses.append(jnp.mean((preds - true_values)**2))
        return jnp.mean(jnp.stack(mses)).item()

    # Okay now go!
    jxkey, subkey = jax.random.split(jxkey)
    ts = create_train_state(run_params.ol_lr, optimizer='adam', jxkey=subkey)
    train_tasks = [next(data_generator) for _ in range(run_params.num_training_tasks)]

    for epoch in range(1, run_params.num_epochs + 1):
        # Use a separate PRNG key to permute image data during shuffling
        jxkey, subkey = jax.random.split(jxkey)
        # Run an optimization step over a training batch
        ts = train_epoch(ts, train_tasks, run_params.num_tasks_per_batch, epoch, subkey)
        # Evaluate on the test set after each training epoch
        test_loss = eval_model_on_grid(ts, grid, test_examples)
        print(f' test epoch: {epoch}, loss: {test_loss:.4f}')


if __name__ == '__main__':
    main(Params())

