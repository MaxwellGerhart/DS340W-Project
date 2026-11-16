"""
Lateral reflection symmetry for the OpenSTARLab event models (DS340W phase 2).

Inspired by TacticAI (Wang et al., Nature Communications 15, 1906, 2024), which
builds the pitch's reflection symmetries into its network so that "the symmetry
does not have to be learnt from data". TacticAI works on tracking data and uses
the dihedral group D2 (flip across the halfway line, flip across the long axis).
Event data here is attack-normalised -- every team attacks the same goal -- so
the halfway-line flip is not a symmetry of the task. The flip across the long
axis (left wing <-> right wing) still is, and that is the group used here: C2,
{identity, mirror}.

How each UIED feature behaves under the mirror (verified against the data):

    start_y     lateral coordinate, stored on 0..105 (the preprocessor applies
                the x/y pitch scales the wrong way round)   ->  105 - start_y
    deltaY      raw Wyscout lateral displacement            ->  -deltaY
    dist2goal   distance to goal centre                     invariant
    angle2goal  pi - atan2(|dy|, dx), uses |dy|             invariant
    distance    hypot(deltaX, deltaY)                       invariant
    everything else (action, delta_T, start_x, deltaX, team, seconds, ...) invariant

Features are min-max normalised before they reach the model, and a reflection
x -> c - x in raw units is still an affine involution in normalised units:
n -> a - n with a = (c - 2*min) / (max - min). No clipping is applied to the
mirrored value, so M(M(x)) == x exactly and frame averaging is exactly
equivariant.

Nothing in event/ is modified; the model class, preprocessing and cost function
are imported and used as published.
"""

import numpy as np
import torch
import torch.nn as nn

# Raw-space reflection x -> c - x for each feature the mirror changes.
MIRROR_CENTRE = {"start_y": 105.0, "deltaY": 0.0}

# Column order of the loader's target vector (UEID_loader.target_features).
TARGETS = ["action", "delta_T", "start_x", "start_y"]


class Mirror:
    """The lateral reflection, expressed on normalised model tensors."""

    def __init__(self, features, min_dict, max_dict, num_actions=9):
        self.num_actions = num_actions

        def coeff(name):
            c = MIRROR_CENTRE[name]
            return (c - 2 * min_dict[name]) / (max_dict[name] - min_dict[name])

        self.in_idx = [i for i, f in enumerate(features) if f in MIRROR_CENTRE]
        self.in_a = torch.tensor([coeff(features[i]) for i in self.in_idx])
        self.y_a = coeff("start_y")
        # output layout: [action logits (num_actions), delta_T, start_x, start_y]
        self.out_y = num_actions + 2
        self.gt_y = TARGETS.index("start_y")

    def inputs(self, X):
        """Mirror a (batch, seq, features) input tensor."""
        X = X.clone()
        a = self.in_a.to(X.device, X.dtype)
        X[..., self.in_idx] = a - X[..., self.in_idx]
        return X

    def outputs(self, out):
        """Mirror a (batch, num_actions + 3) prediction tensor."""
        out = out.clone()
        out[:, self.out_y] = self.y_a - out[:, self.out_y]
        return out

    def targets(self, gt):
        gt = gt.clone()
        gt[:, self.gt_y] = self.y_a - gt[:, self.gt_y]
        return gt


class FrameAveraged(nn.Module):
    """Symmetrise a model over {identity, mirror} by frame averaging.

        FA(x) = 1/2 * ( f(x) + M_out f(M_in x) )

    TacticAI's receiver head does the same thing over its four D2 views. The two
    views go through the network as two separate calls with the same batch
    layout, not as one doubled batch: the published NMSTPP adds a positional
    vector chosen by a sample's *slot in the batch* (issue 7), so an event and
    its mirror image must occupy the same slot for the average to be exactly
    equivariant. Action logits and
    delta_T are invariant, so they are averaged directly; the lateral location
    is mirrored back before averaging. The wrapped model's state_dict keys are
    prefixed with "base.", and `.base` recovers the unwrapped model, whose
    parameter count is unchanged.
    """

    def __init__(self, base, mirror):
        super().__init__()
        self.base = base
        self.mirror = mirror

    def forward(self, X):
        return 0.5 * (self.base(X) + self.mirror.outputs(self.base(self.mirror.inputs(X))))


def random_mirror(X, gt, mirror, p=0.5):
    """Data-augmentation control: mirror a random subset of the batch."""
    flip = torch.rand(X.shape[0], device=X.device) < p
    if flip.any():
        X, gt = X.clone(), gt.clone()
        X[flip] = mirror.inputs(X[flip])
        gt[flip] = mirror.targets(gt[flip])
    return X, gt


class FastUEID:
    """Tensor-backed equivalent of dataloaders.UEID_loader.

    UEID_loader builds its index list with a pure-Python loop and then serves each
    sample through two pandas .loc lookups, which is most of the ~24 minute epoch
    on the full training set (issue 6). This builds the same samples from numpy
    arrays held on the device: identical valid_indices, identical input windows,
    identical targets (asserted in dev/probe_equivariance.py --check-loader).
    """

    def __init__(self, df, seq_len, features, device, target_features=TARGETS):
        df = df.reset_index(drop=True)
        match = df["match_id"].to_numpy()
        period = df["Period"].to_numpy()
        n = len(df) - seq_len
        ok = (match[:n] == match[seq_len:seq_len + n]) & (period[:n] == period[seq_len:seq_len + n])
        self.valid_indices = np.nonzero(ok)[0]
        self.seq_len = seq_len
        self.X = torch.tensor(df[features].to_numpy(dtype=np.float32), device=device)
        self.Y = torch.tensor(df[target_features].to_numpy(dtype=np.float32), device=device)
        self.device = device
        self._offsets = torch.arange(seq_len, device=device)

    def __len__(self):
        return len(self.valid_indices)

    def batches(self, batch_size, shuffle=False, drop_last=False, generator=None):
        """Yield (inputs, targets, end_idx) exactly as DataLoader(UEID_loader) would."""
        idx = torch.tensor(self.valid_indices, device=self.device)
        if shuffle:
            idx = idx[torch.randperm(len(idx), generator=generator).to(self.device)]
        stop = len(idx) - (len(idx) % batch_size if drop_last else 0)
        for s in range(0, stop, batch_size):
            start = idx[s:s + batch_size]
            end = start + self.seq_len
            yield self.X[start[:, None] + self._offsets], self.Y[end], end


@torch.no_grad()
def predict(model, data, batch_size=256):
    """All predictions and targets over a FastUEID split, in order."""
    model.eval()
    outs, gts = [], []
    for X, gt, _ in data.batches(batch_size):
        outs.append(model(X))
        gts.append(gt)
    return torch.cat(outs), torch.cat(gts)


@torch.no_grad()
def equivariance_gap(model, data, mirror, min_dict, max_dict, batch_size=256):
    """How far a model is from respecting the mirror, in interpretable units.

    Compares f(M x) with M f(x). Zero on every row for a frame-averaged model.
    Batches of 256 in file order, as UEID_inference uses: with the unfixed
    positional encoding a prediction depends on its batch slot, so the batch
    layout is part of what is being measured.
        y_gap_m     mean |y_hat(Mx) - mirror(y_hat(x))|, pitch units of start_y
        x_gap_m     mean |x_hat(Mx) - x_hat(x)|,         pitch units of start_x
        dt_gap_s    mean |dT_hat(Mx) - dT_hat(x)|,        seconds
        action_flip share of events whose argmax action changes under the mirror
        action_tv   mean total-variation distance between the two action distributions
    """
    model.eval()
    k = mirror.num_actions
    sums = dict(y_gap_m=0.0, x_gap_m=0.0, dt_gap_s=0.0, action_flip=0.0, action_tv=0.0)
    n = 0
    ry = max_dict["start_y"] - min_dict["start_y"]
    rx = max_dict["start_x"] - min_dict["start_x"]
    lo, hi = min_dict["delta_T"], max_dict["delta_T"]

    def seconds(v):
        return torch.exp(v * (hi - lo) + lo) - 1e-6

    for X, _, _ in data.batches(batch_size):
        a = model(X)
        b = mirror.outputs(model(mirror.inputs(X)))  # mirrored back into x's frame
        sums["y_gap_m"] += ((a[:, k + 2] - b[:, k + 2]).abs() * ry).sum().item()
        sums["x_gap_m"] += ((a[:, k + 1] - b[:, k + 1]).abs() * rx).sum().item()
        sums["dt_gap_s"] += (seconds(a[:, k]) - seconds(b[:, k])).abs().sum().item()
        sums["action_flip"] += (a[:, :k].argmax(1) != b[:, :k].argmax(1)).float().sum().item()
        pa, pb = a[:, :k].softmax(1), b[:, :k].softmax(1)
        sums["action_tv"] += (0.5 * (pa - pb).abs().sum(1)).sum().item()
        n += X.shape[0]
    return {key: v / n for key, v in sums.items()}


METRICS = ["loss", "CEL_action", "RMSE_deltaT", "RMSE_location",
           "ACC_action", "F1_action", "MAE_deltaT", "MAE_x", "MAE_y"]


@torch.no_grad()
def package_metrics(model, data, min_dict, max_dict, config, batch_size=256, mirror_data=None):
    """Test metrics computed the way UEID_inference computes them.

    The package evaluates cost_function on sequential batches of `batch_size`
    and reports the mean of the per-batch values, so F1_action is a per-batch
    macro-F1 averaged over batches, not a global F1. Reproducing that exactly is
    what makes these numbers comparable with Table 6 and REPRODUCTION.md. With
    `mirror_data` set, every input *and* target is mirrored first, i.e. the model
    is scored on the mirror image of the test set.

    Also returns global accuracy and global macro-F1 over all events
    (F1_global), which do not depend on batch composition.
    """
    from event.sports.soccer.utils.UEID_cost_function import cost_function

    model.eval()
    device = data.device
    totals = np.zeros(len(METRICS))
    nb = 0
    preds, labels = [], []
    for X, gt, _ in data.batches(batch_size):
        if mirror_data is not None:
            X, gt = mirror_data.inputs(X), mirror_data.targets(gt)
        out = model(X)
        loss, comps = cost_function(gt, out, min_dict=min_dict, max_dict=max_dict,
                                    device=device, config=config)
        totals += np.array([loss.item()] + [c.item() for c in comps])
        nb += 1
        preds.append(out[:, :config["num_actions"]].argmax(1))
        labels.append(gt[:, 0].long())
    res = dict(zip(METRICS, totals / nb))
    p, t = torch.cat(preds), torch.cat(labels)
    k = config["num_actions"]
    f1s = []
    for c in range(k):
        tp = ((p == c) & (t == c)).sum().item()
        fp = ((p == c) & (t != c)).sum().item()
        fn = ((p != c) & (t == c)).sum().item()
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    res["ACC_global"] = (p == t).float().mean().item()
    res["F1_global"] = float(np.mean(f1s))
    res["F1_per_class"] = f1s
    return res
