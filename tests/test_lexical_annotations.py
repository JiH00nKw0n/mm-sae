import pytest

from mm_sae.data.text import CaptionEditor, load_concepts


@pytest.mark.parametrize(
    "text, concept, expressions",
    [
        ("Buses pass two benches.", 5, ["Buses"]),
        ("Buses pass two benches.", 14, ["benches"]),
        ("A puppy beside two puppies.", 17, ["puppy", "puppies"]),
        ("Wine glasses and knives are on shelves.", 45, ["Wine glasses"]),
        ("Wine glasses and knives are on shelves.", 48, ["knives"]),
        ("Wine glasses and knives are on shelves.", 155, ["shelves"]),
        ("A sofa beside a television.", 62, ["sofa"]),
        ("A sofa beside a television.", 71, ["television"]),
        ("A mirror hangs above a stone floor.", 132, ["mirror"]),
        ("A mirror hangs above a stone floor.", 115, ["stone floor"]),
    ],
)
def test_dictionary_recognizes_forms_and_preserves_exact_source_spans(text, concept, expressions):
    present, spans, status, unavailable = CaptionEditor(load_concepts(None)).analyze(1, text)
    assert concept in present and concept not in unavailable
    assert [text[a:b] for a, b in spans[concept]] == expressions
    assert status == "automatic_character_spans"


def test_dictionary_does_not_guess_ambiguous_terms_or_match_substrings():
    present, _, _, _ = CaptionEditor(load_concepts(None)).analyze(
        1, "He carries luggage and a bat near a cathedral."
    )
    assert not ({0, 16, 26, 32, 38} & set(present))


def test_longest_compound_keeps_other_mentions_of_the_shorter_category():
    text = "A dog beside a hot dog and another dog."
    present, spans, _, unavailable = CaptionEditor(load_concepts(None)).analyze(1, text)
    assert {17, 57} <= set(present) and unavailable == []
    assert [text[a:b] for a, b in spans[17]] == ["dog", "dog"]
    assert [text[a:b] for a, b in spans[57]] == ["hot dog"]
