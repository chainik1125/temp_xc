"""Sanity tests for the quotedens label logic
(experiments/explorations/task_hunt/labels/quotedens_lib.py — B9)."""

import numpy as np
import pytest

from experiments.explorations.task_hunt.labels import quotedens_lib as ql


def test_double_quote_family_fires():
    assert ql.is_quote_sentence('He said, "come in."')
    assert ql.is_quote_sentence('“Come in,” she said.')
    assert ql.is_quote_sentence('„Komm herein“, sagte sie.')
    assert ql.is_quote_sentence('«Entrez», dit-elle.')


def test_single_quotes_and_apostrophes_do_not_fire():
    assert not ql.is_quote_sentence("He said, 'come in.'")
    assert not ql.is_quote_sentence("It was John's house.")
    assert not ql.is_quote_sentence("Plain narration, no dialogue.")


def test_kernel_is_the_frozen_punctint_kernel():
    ev = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8)
    lam = ql.sentence_lambda(ev)
    from experiments.explorations.task_hunt.labels.punctint_lib import (
        sentence_lambda as pl_lambda,
    )
    ref = pl_lambda(ev)
    m = np.isfinite(ref)
    assert (np.isfinite(lam) == m).all()
    assert lam[m] == pytest.approx(ref[m])
    assert np.isnan(lam[:8]).all()
