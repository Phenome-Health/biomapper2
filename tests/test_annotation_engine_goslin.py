from biomapper2.core.annotation_engine import AnnotationEngine


class _FakeBiolink:
    """Descendants: SmallMolecule subtree contains itself; Gene subtree is disjoint."""

    _SUBTREES = {
        "biolink:SmallMolecule": {"biolink:SmallMolecule"},
        "biolink:Gene": {"biolink:Gene"},
        "biolink:Protein": {"biolink:Protein"},
        "biolink:Disease": {"biolink:Disease"},
    }

    def get_descendants(self, category):
        return set(self._SUBTREES.get(category, {category}))


def _engine():
    # _FakeBiolink duck-types the one method the engine touches (get_descendants); a real
    # BiolinkClient would hit the network at construction.
    return AnnotationEngine(biolink_client=_FakeBiolink())  # type: ignore[arg-type]


def test_goslin_registered_in_registry():
    assert "goslin-lipid" in _engine().annotator_registry


def test_small_molecule_route_includes_goslin_and_mw():
    slugs = _engine()._select_annotators("biolink:SmallMolecule")
    assert "goslin-lipid" in slugs
    assert "metabolomics-workbench" in slugs


def test_non_small_molecule_route_excludes_goslin():
    slugs = _engine()._select_annotators("biolink:Disease")
    assert "goslin-lipid" not in slugs
    assert "metabolomics-workbench" not in slugs
