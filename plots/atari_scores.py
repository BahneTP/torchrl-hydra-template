"""Load human and random Atari reference scores for plotting."""

from pathlib import Path

import pandas as pd


def get_atari_reference_scores(game: str) -> tuple[float, float]:
    table = pd.read_csv(Path(__file__).parent / "data" / "Atari-Human.csv")
    normalized_game = "".join(character for character in game.casefold() if character.isalnum())
    table_games = table["Game"].str.casefold().str.replace(r"[^a-z0-9]", "", regex=True)
    rows = table[table_games == normalized_game]
    if len(rows) != 1:
        raise ValueError(f"expected one Atari reference row for {game!r}, found {len(rows)}")
    row = rows.iloc[0]
    return float(row["Random"]), float(row["Human"])
