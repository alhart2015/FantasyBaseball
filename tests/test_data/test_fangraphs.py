import pytest

from fantasy_baseball.data.fangraphs import (
    load_projection_set,
    parse_hitting_csv,
    parse_pitching_csv,
)


class TestParseHittingCsv:
    def test_parses_standard_columns(self, fixtures_dir):
        df = parse_hitting_csv(fixtures_dir / "steamer_hitters.csv")
        assert "name" in df.columns
        assert "hr" in df.columns
        assert "r" in df.columns
        assert "rbi" in df.columns
        assert "sb" in df.columns
        assert "avg" in df.columns
        assert "ab" in df.columns
        assert "h" in df.columns

    def test_correct_row_count(self, fixtures_dir):
        df = parse_hitting_csv(fixtures_dir / "steamer_hitters.csv")
        assert len(df) == 4

    def test_player_type_set_to_hitter(self, fixtures_dir):
        df = parse_hitting_csv(fixtures_dir / "steamer_hitters.csv")
        assert (df["player_type"] == "hitter").all()

    def test_stat_values_correct(self, fixtures_dir):
        df = parse_hitting_csv(fixtures_dir / "steamer_hitters.csv")
        judge = df[df["name"] == "Aaron Judge"].iloc[0]
        assert judge["hr"] == 45
        assert judge["r"] == 110
        assert judge["rbi"] == 120
        assert judge["sb"] == 5
        assert judge["avg"] == pytest.approx(0.291, abs=0.001)

    def test_raises_on_missing_columns(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("Name,Team,G\nFoo,BAR,100\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            parse_hitting_csv(bad_csv)


class TestParsePitchingCsv:
    def test_parses_standard_columns(self, fixtures_dir):
        df = parse_pitching_csv(fixtures_dir / "steamer_pitchers.csv")
        assert "name" in df.columns
        assert "ip" in df.columns
        assert "w" in df.columns
        assert "k" in df.columns
        assert "era" in df.columns
        assert "whip" in df.columns
        assert "sv" in df.columns

    def test_correct_row_count(self, fixtures_dir):
        df = parse_pitching_csv(fixtures_dir / "steamer_pitchers.csv")
        assert len(df) == 3

    def test_player_type_set_to_pitcher(self, fixtures_dir):
        df = parse_pitching_csv(fixtures_dir / "steamer_pitchers.csv")
        assert (df["player_type"] == "pitcher").all()

    def test_strikeouts_mapped_from_SO(self, fixtures_dir):
        df = parse_pitching_csv(fixtures_dir / "steamer_pitchers.csv")
        cole = df[df["name"] == "Gerrit Cole"].iloc[0]
        assert cole["k"] == 240

    def test_earned_runs_available(self, fixtures_dir):
        df = parse_pitching_csv(fixtures_dir / "steamer_pitchers.csv")
        cole = df[df["name"] == "Gerrit Cole"].iloc[0]
        assert cole["er"] == 70

    def test_hits_allowed_mapped(self, fixtures_dir):
        df = parse_pitching_csv(fixtures_dir / "steamer_pitchers.csv")
        cole = df[df["name"] == "Gerrit Cole"].iloc[0]
        assert cole["h_allowed"] == 154

    def test_games_started_mapped_from_GS(self, fixtures_dir):
        """Regression: GS must flow through parse_pitching_csv as `gs` so
        downstream filter_starting_pitchers (upcoming-starts plan) can
        distinguish SPs from RPs. Bug fix: PITCHING_COLUMN_MAP previously
        omitted the GS->gs entry, leaving the source column un-renamed and
        getting dropped by the blend pipeline."""
        df = parse_pitching_csv(fixtures_dir / "steamer_pitchers.csv")
        assert "gs" in df.columns
        cole = df[df["name"] == "Gerrit Cole"].iloc[0]
        assert cole["gs"] == 32
        clase = df[df["name"] == "Emmanuel Clase"].iloc[0]
        assert clase["gs"] == 0  # closer

    def test_parse_succeeds_when_GS_column_absent(self, tmp_path):
        """`gs` is optional — older CSVs without a GS column still parse.
        REQUIRED_PITCHING_COLS does not include `gs`."""
        csv = tmp_path / "no_gs.csv"
        csv.write_text(
            "Name,Team,IP,W,SO,ERA,WHIP,SV,ER,BB,H\n"
            "Some Pitcher,XYZ,180,10,170,3.50,1.20,0,70,55,160\n"
        )
        df = parse_pitching_csv(csv)
        assert "gs" not in df.columns
        assert df.iloc[0]["name"] == "Some Pitcher"


class TestLoadProjectionSet:
    def test_loads_matching_files(self, fixtures_dir):
        hitters, pitchers = load_projection_set(fixtures_dir, "steamer")
        assert len(hitters) == 4
        assert len(pitchers) == 3

    def test_returns_empty_for_missing_system(self, fixtures_dir):
        hitters, pitchers = load_projection_set(fixtures_dir, "nonexistent")
        assert hitters.empty
        assert pitchers.empty
