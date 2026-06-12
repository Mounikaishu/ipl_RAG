import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class IPLChunkPreparer:

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150
        )

    def prepare_chunks(self, documents):
        final_chunks = []

        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get("content", "").strip()
                source = doc.get("source", "unknown")
                metadata = {"source": source}
            else:
                text = doc.page_content.strip()
                metadata = doc.metadata or {}
                source = metadata.get("source", "unknown")

            if not text:
                continue

            lines = text.split("\n")

            if source == "teams":
                # Tables:
                # 1. Team | Short | Home Venue | Captain | Coach | Titles | 2024 Pos
                # 2. Team | Playing Style & Strategy (2 columns)
                for line in lines:
                    line = line.strip()
                    if "|" not in line or "---" in line:
                        continue
                    cols = [c.strip() for c in line.split("|") if c.strip()]
                    if not cols or cols[0].lower() == "team":
                        continue
                    if len(cols) >= 6:
                        team = cols[0]
                        short = cols[1]
                        venue = cols[2]
                        captain = cols[3]
                        coach = cols[4]
                        titles = cols[5]
                        pos = cols[6] if len(cols) > 6 else ""
                        content = f"Team: {team} ({short}), Home Venue: {venue}, Captain: {captain}, Coach: {coach}, Titles Won: {titles}, 2024 Position: {pos}"
                        final_chunks.append(Document(
                            page_content=content,
                            metadata={
                                "team_name": team,
                                "short_name": short,
                                "season": 2024,
                                "section": "team",
                                "source": source
                            }
                        ))
                    elif len(cols) == 2:
                        short = cols[0]
                        strategy = cols[1]
                        content = f"Team Strategy ({short}): Playing Style & Strategy: {strategy}"
                        final_chunks.append(Document(
                            page_content=content,
                            metadata={
                                "team_name": short,
                                "season": 2024,
                                "section": "team",
                                "source": source
                            }
                        ))

            elif source == "batting_stats":
                # Table: Player | Team | Mat | Runs | Avg | SR | 100s | 50s | HS | Role
                for line in lines:
                    line = line.strip()
                    if "|" not in line or "---" in line:
                        continue
                    cols = [c.strip() for c in line.split("|") if c.strip()]
                    if not cols or cols[0].lower() == "player":
                        continue
                    if len(cols) >= 9:
                        player = cols[0]
                        team = cols[1]
                        mat = cols[2]
                        runs = cols[3]
                        avg = cols[4]
                        sr = cols[5]
                        hundreds = cols[6]
                        fifties = cols[7]
                        hs = cols[8]
                        role = cols[9] if len(cols) > 9 else ""
                        content = f"Player: {player}, Team: {team}, Matches: {mat}, Runs: {runs}, Average: {avg}, Strike Rate: {sr}, 100s: {hundreds}, 50s: {fifties}, High Score: {hs}, Role: {role}"
                        final_chunks.append(Document(
                            page_content=content,
                            metadata={
                                "player_name": player,
                                "team": team,
                                "role": role,
                                "section": "batting",
                                "source": source
                            }
                        ))

            elif source == "bowling_stats":
                # Table: Player | Team | Mat | Wkts | Avg | Econ | SR | Best | Type
                for line in lines:
                    line = line.strip()
                    if "|" not in line or "---" in line:
                        continue
                    cols = [c.strip() for c in line.split("|") if c.strip()]
                    if not cols or cols[0].lower() == "player":
                        continue
                    if len(cols) >= 8:
                        player = cols[0]
                        team = cols[1]
                        mat = cols[2]
                        wkts = cols[3]
                        avg = cols[4]
                        econ = cols[5]
                        sr = cols[6]
                        best = cols[7]
                        bowl_type = cols[8] if len(cols) > 8 else ""
                        content = f"Player: {player}, Team: {team}, Matches: {mat}, Wickets: {wkts}, Average: {avg}, Economy: {econ}, Strike Rate: {sr}, Best Figures: {best}, Bowling Type: {bowl_type}"
                        final_chunks.append(Document(
                            page_content=content,
                            metadata={
                                "player_name": player,
                                "team": team,
                                "bowl_type": bowl_type,
                                "section": "bowling",
                                "source": source
                            }
                        ))

            elif source == "h2h_records":
                # Table: Matchup | Total Matches | Team 1 Wins | Team 2 Wins | Tied/NR | Last 5 (→ winner) | High Score | Key Factor
                for line in lines:
                    line = line.strip()
                    if "|" not in line or "---" in line:
                        continue
                    cols = [c.strip() for c in line.split("|") if c.strip()]
                    if not cols or cols[0].lower() == "matchup":
                        continue
                    if len(cols) >= 7:
                        matchup = cols[0]
                        total = cols[1]
                        t1_wins = cols[2]
                        t2_wins = cols[3]
                        tied = cols[4]
                        last5 = cols[5]
                        high_score = cols[6]
                        key_factor = cols[7] if len(cols) > 7 else ""

                        # Extract team names for bidirectional matching
                        teams_split = re.split(r'\s+vs\s+|\s+v\s+|-', matchup.lower())
                        team1 = teams_split[0].strip().upper() if len(teams_split) > 0 else ""
                        team2 = teams_split[1].strip().upper() if len(teams_split) > 1 else ""

                        TEAM_MAP = {
                            "MI": "Mumbai Indians",
                            "CSK": "Chennai Super Kings",
                            "RCB": "Royal Challengers Bengaluru",
                            "KKR": "Kolkata Knight Riders",
                            "DC": "Delhi Capitals",
                            "PBKS": "Punjab Kings",
                            "RR": "Rajasthan Royals",
                            "SRH": "Sunrisers Hyderabad",
                            "LSG": "Lucknow Super Giants",
                            "GT": "Gujarat Titans"
                        }

                        content = f"Matchup: {matchup}, Total Matches: {total}, {team1} Wins: {t1_wins}, {team2} Wins: {t2_wins}, Tied/NR: {tied}, Last 5 Matches Winner: {last5}, High Score: {high_score}, Key Factor: {key_factor}"
                        final_chunks.append(Document(
                            page_content=content,
                            metadata={
                                "team1": team1,
                                "team2": team2,
                                "team1_full": TEAM_MAP.get(team1, team1),
                                "team2_full": TEAM_MAP.get(team2, team2),
                                "section": "h2h",
                                "source": source
                            }
                        ))

            elif source == "venue_reports":
                # Narrative and Table.
                # Table: Venue | City | Capacity | Pitch Type | Avg 1st Innings | Batting/Bowling | Dew Factor | Best Strategy
                is_narrative = False
                narrative_text = []
                venue_name = ""
                city_name = ""

                for line in lines:
                    line_strip = line.strip()
                    if not line_strip:
                        continue

                    if "narrative" in line_strip.lower() or "report" in line_strip.lower():
                        is_narrative = True
                        continue

                    if is_narrative:
                        # Parse Narrative block
                        match = re.match(r'^\*?\*?([A-Za-z0-9\s\.\,\-\'\&]+),\s*([A-Za-z\s]+)\*?\*?$', line_strip)
                        if match:
                            if narrative_text and venue_name:
                                content = f"Venue: {venue_name}, City: {city_name} Narrative Report: {' '.join(narrative_text)}"
                                final_chunks.append(Document(
                                    page_content=content,
                                    metadata={
                                        "venue_name": venue_name,
                                        "city": city_name,
                                        "section": "venue",
                                        "source": source
                                    }
                                ))
                                narrative_text = []
                            venue_name = match.group(1).strip()
                            city_name = match.group(2).strip()
                        else:
                            # If it matches known venues but formatting is slightly different
                            for v in ["Wankhede Stadium", "MA Chidambaram", "M Chinnaswamy", "Rajiv Gandhi Intl. Stadium", "Eden Gardens", "Narendra Modi Stadium", "Sawai Mansingh Stadium", "IS Bindra Stadium"]:
                                if v in line_strip and ("," in line_strip or len(line_strip) < 40):
                                    if narrative_text and venue_name:
                                        content = f"Venue: {venue_name}, City: {city_name} Narrative Report: {' '.join(narrative_text)}"
                                        final_chunks.append(Document(
                                            page_content=content,
                                            metadata={
                                                "venue_name": venue_name,
                                                "city": city_name,
                                                "section": "venue",
                                                "source": source
                                            }
                                        ))
                                        narrative_text = []
                                    parts = line_strip.replace("**", "").split(",")
                                    venue_name = parts[0].strip()
                                    city_name = parts[1].strip() if len(parts) > 1 else ""
                                    break
                            else:
                                if venue_name:
                                    narrative_text.append(line_strip)
                    else:
                        if "|" not in line_strip or "---" in line_strip:
                            continue
                        cols = [c.strip() for c in line_strip.split("|") if c.strip()]
                        if not cols or cols[0].lower() == "venue":
                            continue
                        if len(cols) >= 6:
                            venue = cols[0]
                            city = cols[1]
                            capacity = cols[2]
                            pitch_type = cols[3]
                            avg_1st = cols[4]
                            pref = cols[5]
                            dew = cols[6] if len(cols) > 6 else ""
                            strat = cols[7] if len(cols) > 7 else ""
                            content = f"Venue: {venue}, City: {city}, Capacity: {capacity}, Pitch Type: {pitch_type}, Avg 1st Innings: {avg_1st}, Condition: {pref}, Dew Factor: {dew}, Best Strategy: {strat}"
                            final_chunks.append(Document(
                                page_content=content,
                                metadata={
                                    "venue_name": venue,
                                    "city": city,
                                    "pitch_type": pitch_type,
                                    "section": "venue",
                                    "source": source
                                }
                            ))

                if narrative_text and venue_name:
                    content = f"Venue: {venue_name}, City: {city_name} Narrative Report: {' '.join(narrative_text)}"
                    final_chunks.append(Document(
                        page_content=content,
                        metadata={
                            "venue_name": venue_name,
                            "city": city_name,
                            "section": "venue",
                            "source": source
                        }
                    ))

            elif source == "season_trends":
                # Table: Team | 2019 Pos | 2020 Pos | 2021 Pos | 2022 Pos | 2023 Pos | 2024 Pos | Titles (2019-24)
                years = ["2019", "2020", "2021", "2022", "2023", "2024"]
                for line in lines:
                    line = line.strip()
                    if "|" not in line or "---" in line:
                        continue
                    cols = [c.strip() for c in line.split("|") if c.strip()]
                    if not cols or cols[0].lower() == "team":
                        continue
                    if len(cols) >= 7:
                        team = cols[0]
                        for i, year in enumerate(years):
                            pos = cols[i+1]
                            content = f"Team: {team}, Year: {year}, Standing: {pos}"
                            final_chunks.append(Document(
                                page_content=content,
                                metadata={
                                    "team": team,
                                    "year": int(year),
                                    "section": "season",
                                    "source": source
                                }
                            ))

            elif source == "recent_form":
                # Table: Player | Team | Match 1 | Match 2 | Match 3 | Match 4 | Match 5 | Form Trend | Avg (Last 5)
                for line in lines:
                    line = line.strip()
                    if "|" not in line or "---" in line:
                        continue
                    cols = [c.strip() for c in line.split("|") if c.strip()]
                    if not cols or cols[0].lower() == "player":
                        continue
                    if len(cols) >= 8:
                        player = cols[0]
                        team = cols[1]
                        m1 = cols[2]
                        m2 = cols[3]
                        m3 = cols[4]
                        m4 = cols[5]
                        m5 = cols[6]
                        trend = cols[7]
                        avg = cols[8] if len(cols) > 8 else ""
                        content = f"Player: {player}, Team: {team}, Recent Match Scores/Figures: Match 1: {m1}, Match 2: {m2}, Match 3: {m3}, Match 4: {m4}, Match 5: {m5}, Form Trend: {trend}, Average: {avg}"
                        final_chunks.append(Document(
                            page_content=content,
                            metadata={
                                "player_name": player,
                                "team": team,
                                "season": 2024,
                                "section": "form",
                                "source": source
                            }
                        ))

            elif source == "records":
                # Table: Category | Record | Player/Team | Opponent | Venue | Year
                for line in lines:
                    line = line.strip()
                    if "|" not in line or "---" in line:
                        continue
                    cols = [c.strip() for c in line.split("|") if c.strip()]
                    if not cols or cols[0].lower() == "category":
                        continue
                    if len(cols) >= 5:
                        category = cols[0]
                        record = cols[1]
                        player_team = cols[2]
                        opponent = cols[3]
                        venue = cols[4]
                        year = cols[5] if len(cols) > 5 else ""
                        content = f"Category: {category}, Record Value: {record}, Holder: {player_team}, Opponent: {opponent}, Venue: {venue}, Year: {year}"
                        final_chunks.append(Document(
                            page_content=content,
                            metadata={
                                "category": category,
                                "section": "records",
                                "source": source
                            }
                        ))

            elif source == "conflicting_data":
                # Table: Player / Fact | Primary Source Value | Secondary Source Value | Conflict Type | Expected RAG Behaviour
                for line in lines:
                    line = line.strip()
                    if "|" not in line or "---" in line:
                        continue
                    cols = [c.strip() for c in line.split("|") if c.strip()]
                    if not cols or "primary" in cols[1].lower():
                        continue
                    if len(cols) >= 3:
                        fact = cols[0]
                        primary_val = cols[1]
                        secondary_val = cols[2]

                        player_name = ""
                        if "Virat Kohli" in fact:
                            player_name = "Virat Kohli"
                        elif "Yuzvendra Chahal" in fact:
                            player_name = "Yuzvendra Chahal"
                        elif "MS Dhoni" in fact:
                            player_name = "MS Dhoni"

                        content_primary = f"Fact: {fact}, Value: {primary_val}, Source: Primary Source"
                        content_secondary = f"Fact: {fact}, Value: {secondary_val}, Source: Secondary Source"

                        final_chunks.append(Document(
                            page_content=content_primary,
                            metadata={
                                "player_name": player_name,
                                "fact": fact,
                                "source": "primary",
                                "conflict": True,
                                "section": "validation"
                            }
                        ))
                        final_chunks.append(Document(
                            page_content=content_secondary,
                            metadata={
                                "player_name": player_name,
                                "fact": fact,
                                "source": "secondary",
                                "conflict": True,
                                "section": "validation"
                            }
                        ))
            else:
                chunks = self.text_splitter.split_text(text)
                for chunk in chunks:
                    final_chunks.append(Document(page_content=chunk, metadata=metadata))

        print(f"\nFinal chunks: {len(final_chunks)}")
        return final_chunks