import io
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from pypdf import PdfReader


st.set_page_config(
    page_title="Horse Racing Prediction Assistant",
    page_icon="🐎",
    layout="wide",
)

st.title("🐎 Horse Racing Prediction Assistant")
st.caption(
    "Upload a racecard PDF. The app extracts runners and calculates a transparent "
    "UHIS-based rating, win probability, fair odds, market overlay and recommendation."
)

st.info(
    "This version uses only the racecard PDF. It does not require or use a historical "
    "results CSV. Ratings are transparent racecard-based estimates, not guaranteed outcomes."
)


class UHISEngine:
    def calculate(self, metrics: Dict[str, float]) -> float:
        if not metrics:
            return 0.0
        return round(sum(metrics.values()) / len(metrics), 2)


class ProbabilityEngine:
    def calculate(self, uhis: float, total_race_uhis: float) -> float:
        if total_race_uhis <= 0 or uhis <= 0:
            return 0.0
        return round((uhis / total_race_uhis) * 100, 2)


class FairOddsEngine:
    def calculate(self, probability: float) -> float:
        if probability <= 0:
            return 0.0
        return round(100 / probability, 2)


class OverlayEngine:
    def calculate(self, fair_odds: float, market_odds: Optional[float]) -> float:
        if not fair_odds or not market_odds or fair_odds <= 0 or market_odds <= 0:
            return 0.0
        return round(((market_odds - fair_odds) / fair_odds) * 100, 2)


class EESEngine:
    def __init__(
        self,
        uhis_weight: float = 0.60,
        market_weight: float = 0.25,
        draw_weight: float = 0.15,
    ):
        total = uhis_weight + market_weight + draw_weight
        self.uhis_weight = uhis_weight / total
        self.market_weight = market_weight / total
        self.draw_weight = draw_weight / total

    def calculate(self, uhis: float, market: float, draw: float) -> float:
        score = (
            uhis * self.uhis_weight
            + market * self.market_weight
            + draw * self.draw_weight
        )
        return round(score, 2)


class RecommendationEngine:
    def __init__(
        self,
        best_bet_ees: float = 86.0,
        value_overlay: float = 20.0,
        place_bet_ees: float = 78.0,
    ):
        self.best_bet_ees = best_bet_ees
        self.value_overlay = value_overlay
        self.place_bet_ees = place_bet_ees

    def generate(self, ees: float, overlay: float) -> str:
        if ees >= self.best_bet_ees and overlay >= self.value_overlay:
            return "BEST VALUE BET"
        if ees >= self.best_bet_ees:
            return "BEST BET"
        if overlay >= self.value_overlay:
            return "VALUE BET"
        if ees >= self.place_bet_ees:
            return "PLACE BET"
        return "WATCH"


class RacecardParser:
    RACE_HEADER_PATTERN = re.compile(
        r"(?i)\bRACE\s+(?P<race_number>\d{1,2})\b(?P<description>[^\n]*)"
    )
    DISTANCE_PATTERN = re.compile(r"(?i)(?P<distance>\d{3,4})\s*m\b")
    RUNNER_PATTERN = re.compile(
        r"^\s*"
        r"(?P<number>\d{1,2})\s+"
        r"(?:\((?P<draw>\d{1,2})\)\s+)?"
        r"(?P<horse>[A-Za-z][A-Za-z\s'\-\.]{1,35}?)\s+"
        r"(?P<weight>\d{1,2}(?:\.\d)?)\s+"
        r"(?P<remainder>.+?)\s*$"
    )
    ODDS_PATTERN = re.compile(
        r"^(?P<value>\d+\s*/\s*\d+|\d+(?:\.\d+)?)(?:\s*[Ff])?$"
    )

    def parse(self, text: str) -> Dict[str, List[dict]]:
        races: Dict[str, List[dict]] = {}
        headers = list(self.RACE_HEADER_PATTERN.finditer(text))

        if not headers:
            return races

        for index, header in enumerate(headers):
            race_number = header.group("race_number")
            description = header.group("description")

            distance_match = self.DISTANCE_PATTERN.search(description)
            distance = int(distance_match.group("distance")) if distance_match else 0

            start = header.end()
            end = headers[index + 1].start() if index + 1 < len(headers) else len(text)

            runners = self._parse_runners(text[start:end], distance)

            if runners:
                title = f"Race {race_number}"
                if distance:
                    title += f" - {distance}m"
                races[title] = runners

        return races

    def _parse_runners(self, race_block: str, distance: int) -> List[dict]:
        runners = []
        seen = set()

        for line in race_block.splitlines():
            match = self.RUNNER_PATTERN.match(line)
            if not match:
                continue

            horse = re.sub(r"\s+", " ", match.group("horse")).strip().title()
            if horse in seen or len(horse) < 3:
                continue

            draw = int(match.group("draw")) if match.group("draw") else 0
            weight = float(match.group("weight"))
            jockey, trainer, market_odds = self._split_details(match.group("remainder"))

            runners.append(
                {
                    "horse": horse,
                    "draw": draw,
                    "weight": weight,
                    "jockey": jockey,
                    "trainer": trainer,
                    "market_odds": market_odds,
                    "distance": distance,
                }
            )
            seen.add(horse)

        return runners

    def _split_details(self, remainder: str) -> Tuple[str, str, Optional[float]]:
        tokens = remainder.split()
        odds = None

        if tokens and self.ODDS_PATTERN.match(tokens[-1]):
            odds = self._parse_odds(tokens[-1])
            tokens = tokens[:-1]

        if len(tokens) >= 4:
            jockey = " ".join(tokens[:-2])
            trainer = " ".join(tokens[-2:])
        elif len(tokens) == 3:
            jockey = " ".join(tokens[:2])
            trainer = tokens[-1]
        elif len(tokens) == 2:
            jockey, trainer = tokens
        elif len(tokens) == 1:
            jockey, trainer = tokens[0], ""
        else:
            jockey, trainer = "", ""

        return jockey.strip(), trainer.strip(), odds

    @staticmethod
    def _parse_odds(odds_text: str) -> Optional[float]:
        try:
            if "/" in odds_text:
                numerator, denominator = odds_text.split("/")
                return round((float(numerator) / float(denominator)) + 1.0, 2)

            value = float(odds_text)
            return round(value if value > 1 else value + 1, 2)
        except (ValueError, ZeroDivisionError):
            return None


def build_runner_profile(
    horse: str,
    draw: int,
    weight: float,
    jockey: str,
    trainer: str,
    market_odds: Optional[float],
    distance: int,
) -> dict:
    market_score = 70.0
    if market_odds:
        market_score = max(50.0, min(95.0, 100.0 - (market_odds * 2.0)))

    if draw:
        draw_penalty = (draw - 1) * (1.8 if 0 < distance <= 1400 else 1.0)
        draw_score = max(60.0, min(95.0, 92.0 - draw_penalty))
    else:
        draw_score = 78.0

    weight_score = max(60.0, min(94.0, 85.0 + ((weight - 57.0) * 1.2)))

    metrics = {
        "market": market_score,
        "draw": draw_score,
        "weight": weight_score,
        "jockey": 75.0,
        "trainer": 75.0,
        "distance": 75.0,
        "track": 75.0,
        "going": 75.0,
        "pace": (draw_score + market_score) / 2,
    }

    uhis = UHISEngine().calculate(metrics)

    return {
        "horse": horse,
        "jockey": jockey or "Not extracted",
        "trainer": trainer or "Not extracted",
        "draw": draw,
        "weight": weight,
        "market_odds": market_odds,
        "uhis": uhis,
        "market_score": round(market_score, 1),
        "draw_score": round(draw_score, 1),
        "weight_score": round(weight_score, 1),
    }


def analyze_meeting(
    extracted_text: str,
    normalise_probabilities: bool = True,
) -> Dict[str, pd.DataFrame]:
    parser = RacecardParser()
    parsed_races = parser.parse(extracted_text)

    if not parsed_races:
        return {}

    uhis_engine = UHISEngine()
    probability_engine = ProbabilityEngine()
    fair_odds_engine = FairOddsEngine()
    overlay_engine = OverlayEngine()
    ees_engine = EESEngine()
    recommendation_engine = RecommendationEngine()

    analyzed_races = {}

    for race_name, runners in parsed_races.items():
        rows = []

        for runner in runners:
            profile = build_runner_profile(
                horse=runner["horse"],
                draw=runner["draw"],
                weight=runner["weight"],
                jockey=runner["jockey"],
                trainer=runner["trainer"],
                market_odds=runner["market_odds"],
                distance=runner["distance"],
            )

            metrics = {
                "market": profile["market_score"],
                "draw": profile["draw_score"],
                "weight": profile["weight_score"],
                "jockey": 75.0,
                "trainer": 75.0,
                "distance": 75.0,
                "track": 75.0,
                "going": 75.0,
                "pace": (profile["draw_score"] + profile["market_score"]) / 2,
            }

            uhis = uhis_engine.calculate(metrics)

            rows.append(
                {
                    "Horse": profile["horse"],
                    "Draw": profile["draw"] or "N/A",
                    "Weight (kg)": profile["weight"],
                    "Jockey": profile["jockey"],
                    "Trainer": profile["trainer"],
                    "Market odds": profile["market_odds"] or "N/A",
                    "Market rating": profile["market_score"],
                    "Draw rating": profile["draw_score"],
                    "Weight rating": profile["weight_score"],
                    "UHIS": uhis,
                }
            )

        dataframe = pd.DataFrame(rows)
        if dataframe.empty:
            continue

        total_uhis = dataframe["UHIS"].sum()

        dataframe["Win probability %"] = dataframe["UHIS"].apply(
            lambda value: probability_engine.calculate(value, total_uhis)
        )

        if normalise_probabilities:
            probability_total = dataframe["Win probability %"].sum()
            if probability_total > 0:
                dataframe["Win probability %"] = (
                    dataframe["Win probability %"] / probability_total * 100
                ).round(2)

        dataframe["Fair odds"] = dataframe["Win probability %"].apply(
            fair_odds_engine.calculate
        )

        def calculate_overlay(row):
            market_odds = row["Market odds"]
            if not isinstance(market_odds, (int, float)) or pd.isna(market_odds):
                return 0.0
            return overlay_engine.calculate(row["Fair odds"], float(market_odds))

        dataframe["Overlay %"] = dataframe.apply(calculate_overlay, axis=1)

        dataframe["EES"] = dataframe.apply(
            lambda row: ees_engine.calculate(
                row["UHIS"], row["Market rating"], row["Draw rating"]
            ),
            axis=1,
        )

        dataframe["Recommendation"] = dataframe.apply(
            lambda row: recommendation_engine.generate(row["EES"], row["Overlay %"]),
            axis=1,
        )

        analyzed_races[race_name] = (
            dataframe.sort_values("EES", ascending=False).reset_index(drop=True)
        )

    return analyzed_races


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


racecard_file = st.file_uploader("Upload racecard PDF", type=["pdf"])

normalise = st.checkbox(
    "Normalise probabilities so each race totals 100%",
    value=True,
)

if racecard_file:
    try:
        with st.spinner("Reading and analysing racecard..."):
            extracted_text = extract_text_from_pdf(racecard_file)
            races = analyze_meeting(
                extracted_text,
                normalise_probabilities=normalise,
            )

        if not races:
            st.warning(
                "No runners could be extracted. The PDF may be scanned/image-based, "
                "protected, or use an unexpected layout. Try the manual text option below."
            )
        else:
            st.success(f"Analysed {len(races)} race(s).")

            for race_name, dataframe in races.items():
                with st.expander(race_name, expanded=True):
                    st.dataframe(dataframe, use_container_width=True)

                    buffer = io.StringIO()
                    dataframe.to_csv(buffer, index=False)

                    st.download_button(
                        "Download race analysis CSV",
                        data=buffer.getvalue(),
                        file_name=f"{race_name.replace(' ', '_')}.csv",
                        mime="text/csv",
                    )

    except Exception as error:
        st.error(f"Could not process the PDF: {error}")

st.divider()
st.subheader("Manual racecard text")

manual_text = st.text_area(
    "Paste racecard text",
    height=220,
    placeholder=(
        "RACE 1 - 1200M\n"
        "1 (3) CAPE CRUSADER 60.0 R Fourie J Snaith 4/1\n"
        "2 (5) HIGHVELD STAR 58.5 G Lerena M de Kock 5.50"
    ),
)

if st.button("Analyse pasted racecard"):
    if not manual_text.strip():
        st.warning("Please paste racecard text first.")
    else:
        races = analyze_meeting(
            manual_text,
            normalise_probabilities=normalise,
        )

        if not races:
            st.warning("No runners could be parsed from the pasted text.")
        else:
            for race_name, dataframe in races.items():
                st.subheader(race_name)
                st.dataframe(dataframe, use_container_width=True)

st.divider()
st.caption(
    "For personal research and entertainment only. This app does not guarantee racing "
    "outcomes. Its ratings are based on racecard information and should be backtested "
    "before being used for wagering."
)
