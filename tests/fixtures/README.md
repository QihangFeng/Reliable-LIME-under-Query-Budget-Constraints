# Original notebook reference

`notebook_reference.json` records four deterministic runs of the original
`coarse2fine_LIME.ipynb` at commit
`3ea170646e0a348d585bb154e24f084bf10786ee`, before its implementation was
extracted into the Python package. It uses the synthetic 12 by 12 RGB image,
nine fixed superpixels, and three-class classifier in `tests/_fixtures.py`.

The cases cover vanilla and coarse-to-fine LIME at budget 64 with seed 0 and
budget 200 with seed 7. The original class and function definitions were
evaluated in isolation; notebook experiment cells and pretrained model loading
were not executed. The JSON includes the notebook hash, dependency versions,
parameters, fitted predictions and weights, coefficients, explanation masks,
metrics, and fingerprints of all classifier inputs. Runtime is excluded.

These references preserve the existing coarse-to-fine accounting behavior:
it reports the configured budget, evaluates one fewer perturbed image, and
fits its final surrogate with two fewer rows. Fixing that behavior is a separate
algorithm change that requires new reference results. Deletion-test predictions
are separate evaluation work and are excluded from the recorded explainer count.

Tests load the saved references directly. They require neither Git nor network
access, model weights, or a GPU.
