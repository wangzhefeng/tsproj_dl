"""模型与朴素基线共用指标；保留 dtype、mask 与 horizon 宏平均合同。"""
import numpy as np


def forecast_metrics(prediction, target, valid, freq_seconds):
    if prediction.shape != target.shape or valid.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 1:
        raise ValueError("指标需要相同的 [B,H,1] shape")
    if not valid.any() or not np.isfinite(prediction).all():
        raise ValueError("预测无效或无有效目标")
    residual = prediction - target
    summary = {"micro_mae": float(np.abs(residual[valid]).mean()),
               "micro_rmse": float(np.sqrt(np.square(residual[valid]).mean())), "per_horizon": []}
    for k in range(prediction.shape[1]):
        r = residual[:, k][valid[:, k]]
        summary["per_horizon"].append({"horizon": k + 1, "count": len(r),
                                       "mae": float(np.abs(r).mean()) if len(r) else None,
                                       "rmse": float(np.sqrt(np.square(r).mean())) if len(r) else None})
    complete = valid[:, :, 0].all(axis=1)
    if complete.any():
        p, t = prediction[complete, :, 0], target[complete, :, 0]
        summary["peak_amplitude_mae"] = float(np.abs(p.max(axis=1) - t.max(axis=1)).mean())
        summary["peak_time_mae_seconds"] = float(np.abs(p.argmax(axis=1) - t.argmax(axis=1)).mean() * freq_seconds)
    else:
        summary["peak_amplitude_mae"] = summary["peak_time_mae_seconds"] = None
    ramp_valid = valid[:, 1:, 0] & valid[:, :-1, 0]
    ramp_error = np.diff(prediction[:, :, 0], axis=1) - np.diff(target[:, :, 0], axis=1)
    summary["ramp_mae"] = float(np.abs(ramp_error[ramp_valid]).mean()) if ramp_valid.any() else None
    summary["complete_window_count"] = int(complete.sum())
    summary["ramp_count"] = int(ramp_valid.sum())
    return summary


def latent_metrics(errors, quality, dynamic_groups, stats):
    groups = {}
    for g, name in enumerate(dynamic_groups):
        q = quality[:, :, g]
        valid = q > 0
        counts = valid.sum(axis=0)
        weight = np.minimum(100., 1 / (stats[name + "__variance"] + 0.01))
        weighted = (errors[:, :, g] * weight).mean(axis=-1)
        per_h = np.divide((q * weighted).sum(axis=0), counts, out=np.zeros(q.shape[1]), where=counts > 0)
        groups[name] = {"valid_count": int(valid.sum()), "per_horizon_count": counts.tolist(),
                        "quality_mean": float(q.mean()), "quality_zero_rate": float((q == 0).mean()),
                        "weighted_horizon_macro_mse": float(per_h[counts > 0].mean()) if (counts > 0).any() else None,
                        "raw_coordinate_mse": float(errors[:, :, g][valid].mean()) if valid.any() else None,
                        "training_variance": stats[name + "__variance"].tolist()}
    return {"coordinate": "nonphysical_canonical_latent", "groups": groups}
