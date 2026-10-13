"""Q4: the authorization gate, the protocol check and the compute ledger."""

from __future__ import annotations

import pytest

from nuisance import provenance as P


@pytest.mark.parametrize(
    "text,expected",
    [("0 only", {0}), ("1", {1}), ("2–4", {2, 3, 4}), ("1-2", {1, 2}), ("1, 3", {1, 3}), ("1, 3–4 only", {1, 3, 4})],
)
def test_parse_stages(text, expected):
    assert P.parse_stages(text) == expected


@pytest.mark.parametrize("text", ["stages 1-5 on hold", "all", "", "1 and 2"])
def test_unreadable_authorizations_are_refused(text):
    with pytest.raises(ValueError):
        P.parse_stages(text)


def test_authorized_stages_reads_only_dated_rows(tmp_path):
    log = tmp_path / "LOG.md"
    log.write_text(
        "# Log\n\n## Authorizations\n\n| Date | From | Stage(s) | Wording |\n|---|---|---|---|\n"
        "| — | — | — | — |\n| 2026-10-12 | lead | 0 only | paraphrase |\n\n## Deviations\n\n| 2026-10-13 | x | 1 | y |\n",
        encoding="utf-8",
    )
    assert P.authorized_stages(log) == {0}


def test_real_log_parses_and_protocol_is_frozen():
    P.authorized_stages()  # raises if an entry is unreadable
    assert len(P.check_prereg()) == 64
