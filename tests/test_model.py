import pytest
import torch


def test_predict_logits(image_batch, model):
    logits = model.predict_logits(image_batch)

    assert logits.shape[0] == image_batch.shape[0]
    assert logits.shape[1] == model.num_classes()


def test_predict_proba(image_batch, model):
    proba = model.predict_proba(image_batch)

    assert proba.shape[0] == image_batch.shape[0]
    assert proba.shape[1] == model.num_classes()
    assert torch.allclose(proba.sum(dim=-1), torch.tensor(1, dtype=torch.get_default_dtype()))


def tets_predict_class(image_batch, model):
    classes = model.predict_class(image_batch)

    assert classes.shape[0] == image_batch.shape[0]
    assert classes.ndim == 1
