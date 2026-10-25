# Dataset integration contract

## Events: UTF-8 CSV

All 14 columns are required, in any order. Every file may contain several matches, but an analysis run selects exactly one. IDs may be strings. All event IDs must be unique within each match.

| Column | Meaning |
| --- | --- |
| match_id | Match identity |
| event_id | Stable event identity, preserved from source when possible |
| period | Integer 1 to 4 |
| time_seconds | Monotonic match elapsed seconds from kickoff, excluding halftime breaks; 0 to 12000 |
| team | Team name or stable ID; exactly two teams per match |
| player | Actor identity, required |
| receiver | Recipient identity; required for completed passes, otherwise empty |
| type | pass, carry, shot, goal, recovery, pressure, substitution, red_card |
| outcome | success, fail, or na |
| x, y | Event start position, normalized to [0, 100] |
| end_x, end_y | Event end position, normalized to [0, 100] |

**Event coordinates:** each event is in its acting team's attacking direction, always toward x=100. Use the same y convention throughout; mirroring x and y together is a valid full rotation. The standard pitch assumed for metric conversion is 105 by 68 meters. For non-movement context events, use a documented placeholder coordinate such as 50,50. Context events are excluded from on-ball counts.

**Clock:** convert provider period clocks into a single strictly interpretable elapsed time axis. Preserve stoppage time and exclude halftime breaks. If period one lasts 47 minutes, period two starts at elapsed minute 47, not 45. Record the conversion in adapter metadata. Windows use [start, end), so a boundary event belongs to exactly one window. Timing is independent of wall-clock dates.

**Goals:** a `goal` is a successful scoring attempt and counts once toward shots and the score. Do not also emit a duplicate `shot` for the same attempt. Convert own goals to scoring-team goal rows, preserving source provenance in adapter metadata. Without complete goal events, score context will be incomplete.

**Pressure:** only use recorded pressure events. Do not infer missing pressure events. The current schema cannot express per-provider event-type coverage. If the entire match has no pressure records, pressure-per-100 stays unavailable rather than being reported as a genuine zero. Explicit per-provider capability declarations are still an integration task before using real provider data.

**Context:** substitutions and red cards are optional observations. Their absence does not prove no such events occurred. Full substitution identity, lineup state and player roles are not yet modeled. Fatigue is never inferred.

Example:

```csv
match_id,event_id,period,time_seconds,team,player,receiver,type,outcome,x,y,end_x,end_y
match-001,e001,1,15,Home,H06,H08,pass,success,30,45,46,52
match-001,e002,1,19,Away,A04,,recovery,success,54,48,54,48
```

This tiny example demonstrates the format; it is too sparse for useful analysis.

## Optional tracking: JSON array

Tracking uses **one common coordinate frame**, with home attacking +x throughout, including after halftime normalization. Both teams' positions are in this same frame. This differs from the possession-oriented event coordinates. `home_team` and `away_team` identify the same two teams for every snapshot.

```json
[
  {
    "match_id": "match-001",
    "time_seconds": 30,
    "home_team": "Home",
    "away_team": "Away",
    "players": [
      {"team": "Home", "player": "H01", "x": 6.0, "y": 50.0},
      {"team": "Away", "player": "A01", "x": 94.0, "y": 50.0}
    ]
  }
]
```

The illustration omits outfield players. **Each real snapshot must include 7 to 11 players per team**, unique within each team/frame. Timestamps must be unique, finite and aligned with the event clock. Unevenly sampled snapshots currently receive equal weight; resample tracking to a regular cadence before analysis. Occluded or partial freeze-frame data is not full tracking. The validator's player-count check cannot establish complete coverage by itself.

Velocities, ball location and lineup roles are not required in this version. Goalkeepers are excluded from width/depth by selecting the player nearest the team's own goal line. This heuristic should be replaced by provider lineup roles before evaluating compactness scientifically.

## Provider adapter checklist

1. Obtain the exact dataset, license and citation.
2. Keep raw data immutable under `data/raw/`; record SHA-256 and source URLs.
3. Map IDs, timestamps, periods, teams, players and event labels.
4. Normalize event coordinates to the acting team's +x direction.
5. Normalize tracking to the common home-attacks-+x frame.
6. Deduplicate goal/shot records and account for own goals and score context.
7. Declare which event types and tracking fields are actually observed.
8. Export normalized CSV/JSON, plus adapter metadata.
9. Manually verify a pass, a shot, a goal, halftime orientation, and a substitution.
10. Add a small fixture test for that provider adapter and run the project tests.

The included `data/demo/metadata.json` labels the demo as synthetic. Never relabel synthetic records as observations from real matches.
