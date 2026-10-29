from perception.detector.scenarios import scenario_datasets


class FakeDataset:
    def __init__(self) -> None:
        self.images = [
            {
                "id": 1,
                "set": "A",
                "camera": "table",
                "variant": "native",
                "products": [1, 2],
                "defect_types": [],
                "build_states": [],
                "has_trap": False,
            },
            {
                "id": 2,
                "set": "G",
                "camera": "side",
                "variant": "native",
                "products": [2, 3],
                "defect_types": ["crack"],
                "build_states": ["high"],
                "has_trap": True,
            },
            {
                "id": 3,
                "set": "G",
                "camera": "table",
                "variant": "cosmos",
                "products": [3],
                "defect_types": ["crack", "scratch"],
                "build_states": ["tilted"],
                "has_trap": True,
            },
        ]
        self.categories = ("F01", "M1_01")
        self.annotations = {1: [{"id": 11}], 2: [{"id": 12}], 3: [{"id": 13}]}
        self.processor = object()

    def __len__(self) -> int:
        return len(self.images)


def _ids(dataset: FakeDataset) -> list[int]:
    return [image["id"] for image in dataset.images]


def test_scenario_datasets_slice_every_non_empty_detector_dimension() -> None:
    source = FakeDataset()

    slices = scenario_datasets(source)

    assert _ids(slices["set=A"]) == [1]
    assert _ids(slices["camera=table"]) == [1, 3]
    assert _ids(slices["variant=cosmos"]) == [3]
    assert _ids(slices["product=2"]) == [1, 2]
    assert _ids(slices["defect_type=crack"]) == [2, 3]
    assert _ids(slices["build_state=high"]) == [2]
    assert _ids(slices["trap=present"]) == [2, 3]
    assert _ids(slices["trap=absent"]) == [1]
    assert all(len(dataset) for dataset in slices.values())


def test_scenario_slices_reuse_categories_annotations_and_leave_source_unchanged() -> None:
    source = FakeDataset()

    slices = scenario_datasets(source)

    assert _ids(source) == [1, 2, 3]
    assert slices["set=G"].categories is source.categories
    assert slices["set=G"].annotations is source.annotations
    assert slices["set=G"].processor is source.processor
