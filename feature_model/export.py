"""Train the feature route on all labeled rooms and export a submission ONNX.

    python -m feature_model.export --model logreg --output submissions/feature_logreg.onnx
    python -m feature_model.export --model lgbm   --output submissions/feature_lgbm.onnx

The feature extractor is exported from PyTorch. A logistic regression is
folded into the same PyTorch graph; LightGBM is converted with onnxmltools and
merged behind the extractor. Both output log-probabilities as `logits`.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from torch import nn

from common.export import RawInputWrapper, check_onnx, export_onnx, raw_samples
from feature_model.extract import load_features
from feature_model.features import FeatureExtractor
from feature_model.train import make_classifier, select_columns


class LinearHead(nn.Module):
    """Standardise the selected features and apply a fitted linear model."""

    def __init__(self, extractor, cols, scaler, logreg):
        super().__init__()
        self.extractor = extractor
        self.register_buffer("cols", torch.as_tensor(cols, dtype=torch.long))
        self.register_buffer("mean", torch.as_tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scaler.scale_, dtype=torch.float32))
        self.linear = nn.Linear(len(cols), logreg.coef_.shape[0])
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(logreg.coef_, dtype=torch.float32))
            self.linear.bias.copy_(torch.as_tensor(logreg.intercept_, dtype=torch.float32))

    def forward(self, h, ati):
        x = self.extractor(h, ati)[:, self.cols]
        return torch.log_softmax(self.linear((x - self.mean) / self.scale), dim=-1)


class SelectedFeatures(nn.Module):
    def __init__(self, extractor, cols):
        super().__init__()
        self.extractor = extractor
        self.register_buffer("cols", torch.as_tensor(cols, dtype=torch.long))

    def forward(self, h, ati):
        return self.extractor(h, ati)[:, self.cols]


def export_lgbm(extractor, cols, clf, output):
    """Export features with PyTorch, convert LightGBM, and merge the two graphs."""
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    feat_path = output.with_suffix(".features.onnx")
    samples = raw_samples(1)
    # Keep the batch axis: the converted LightGBM graph expects (1, F).
    wrapper = RawInputWrapper(SelectedFeatures(extractor, cols), squeeze=False).eval()
    torch.onnx.export(wrapper, (torch.from_numpy(samples[0][0]), torch.from_numpy(samples[0][1])),
                      str(feat_path), input_names=["frames", "intervals"],
                      output_names=["features"], opset_version=18, external_data=False)
    feat_model = onnx.load(str(feat_path))
    feat_path.unlink()

    # onnxmltools supports opset <= 15; lift the result to the extractor's opset 18.
    tree_model = convert_lightgbm(
        clf, initial_types=[("features", FloatTensorType([1, len(cols)]))],
        target_opset=15, zipmap=False,
    )
    tree_model = onnx.version_converter.convert_version(tree_model, 18)
    tree_model = onnx.compose.add_prefix(tree_model, prefix="gbm_")
    tree_model.ir_version = feat_model.ir_version
    merged = onnx.compose.merge_models(feat_model, tree_model, io_map=[("features", "gbm_features")],
                                       outputs=["gbm_probabilities"])
    # probabilities (1, 5) -> log-probabilities (5,) named "logits"
    g = merged.graph
    g.initializer.extend([
        helper.make_tensor("lgbm_eps", TensorProto.FLOAT, [], [1e-7]),
        helper.make_tensor("lgbm_shape", TensorProto.INT64, [1], [5]),
    ])
    g.node.extend([
        helper.make_node("Add", ["gbm_probabilities", "lgbm_eps"], ["lgbm_p"]),
        helper.make_node("Log", ["lgbm_p"], ["lgbm_logp"]),
        helper.make_node("Reshape", ["lgbm_logp", "lgbm_shape"], ["logits"]),
    ])
    del g.output[:]
    g.output.append(helper.make_tensor_value_info("logits", TensorProto.FLOAT, [5]))
    onnx.save(merged, str(output))

    def reference(frames, intervals):
        x = wrapper(frames, intervals).detach().numpy()
        return torch.from_numpy(np.log(clf.predict_proba(x)[0] + 1e-7).astype(np.float32))

    return check_onnx(output, reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="logreg", choices=["lgbm", "logreg"])
    parser.add_argument("--exclude", default="")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path("submissions") / f"feature_{args.model}.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)

    data = load_features("labeled")
    cols = select_columns(data["names"], args.exclude)
    clf = make_classifier(args.model).fit(data["X"][:, cols], data["label"])
    print(f"trained {args.model} on {len(data['label'])} windows, {len(cols)} features")
    export_classifier(args.model, clf, cols, output)


def export_classifier(kind, clf, cols, output):
    """Write a fitted classifier (on feature columns `cols`) as a submission ONNX."""
    extractor = FeatureExtractor().eval()
    if kind == "logreg":
        scaler, logreg = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
        return export_onnx(LinearHead(extractor, cols, scaler, logreg).eval(), output)
    return export_lgbm(extractor, cols, clf, output)


if __name__ == "__main__":
    main()
