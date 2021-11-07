import pytest
import torch


def test_predict_logits(image_batch, model):
    logits = model.predict_logits(image_batch)

    assert logits.shape[0] == image_batch.shape[0]
    assert logits.ndim == 1


def test_predict_proba(image_batch, model):
    logits = model.predict_logits(image_batch)
    proba = model.predict_proba(logits)

    assert proba.shape[0] == image_batch.shape[0]
    assert proba.ndim == 1


def tets_predict_class(image_batch, model):
    logits = model.predict_logits(image_batch)
    classes = model.predict_class(logits)

    assert classes.shape[0] == image_batch.shape[0]
    assert classes.ndim == 1
