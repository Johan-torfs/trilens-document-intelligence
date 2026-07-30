import numpy as np

from app.repositories.vector_repository import VectorRepository


def test_saves_and_loads_vector(
    vector_repository: VectorRepository,
) -> None:
    vector = np.array(
        [0.1, 0.2, 0.3, 0.4],
        dtype=np.float32,
    )

    stored_path = vector_repository.save(
        artifact_id="invoice_001_clip",
        vector=vector,
    )

    loaded_vector = vector_repository.load(
        artifact_id="invoice_001_clip"
    )

    assert stored_path.exists()
    np.testing.assert_array_equal(
        loaded_vector,
        vector,
    )