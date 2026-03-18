# Fantasy Baseball Tools — Setup Guide

Step-by-step instructions for getting the Fantasy Baseball toolkit running on your machine. Works on Mac and Windows.

---

## 1. Install Python

**Mac:**
1. Open **Terminal** (press Cmd+Space, type "Terminal", press Enter)
2. Type `python3 --version` and press Enter
3. If it says "command not found", go to https://www.python.org/downloads/, download the macOS installer, and run it
4. After installing, close and reopen Terminal, then try `python3 --version` again — you should see something like `Python 3.12.x`

**Windows:**
1. Open the **Microsoft Store**, search for "Python 3.12", and click **Install**
   - Alternatively: download from https://www.python.org/downloads/ and run the installer. **Check the box that says "Add Python to PATH"** — this is important!
2. Open **Command Prompt** (press Win+R, type `cmd`, press Enter)
3. Type `python --version` and press Enter — you should see something like `Python 3.12.x`

---

## 2. Install Git

**Mac:**
1. In Terminal, type `git --version` and press Enter
2. If it's not installed, a popup will ask you to install "Xcode Command Line Tools" — click **Install** and wait for it to finish

**Windows:**
1. Go to https://git-scm.com/download/win and download the installer
2. Run it — accept all the default options
3. **Close and reopen Command Prompt** after installing
4. Type `git --version` to verify

---

## 3. Clone the repository

Open Terminal (Mac) or Command Prompt (Windows) and run:

```
git clone https://github.com/alhart2015/FantasyBaseball.git
cd FantasyBaseball
```

This downloads the code to a `FantasyBaseball` folder.

---

## 4. Install dependencies

**Mac:**
```
python3 -m pip install -e ".[dev]"
```

**Windows:**
```
python -m pip install -e ".[dev]"
```

This installs all the libraries the tool needs. It may take a minute.

---

## 5. Set up Yahoo authentication

Hart will send you two values: a **consumer key** and a **consumer secret**. These let the tool connect to Yahoo Fantasy.

Create the file `config/oauth.json` with this exact content (replace the placeholder values with what Hart sends you):

```json
{
    "consumer_key": "PASTE_CONSUMER_KEY_HERE",
    "consumer_secret": "PASTE_CONSUMER_SECRET_HERE"
}
```

**How to create the file:**
- Open any text editor (TextEdit on Mac, Notepad on Windows)
- Paste the JSON above with your real values
- Save it as `oauth.json` inside the `config` folder in the `FantasyBaseball` directory
- **Important (Mac):** If using TextEdit, go to Format > Make Plain Text before saving
- **Important (Windows):** In Notepad's Save dialog, change "Save as type" to "All Files" so it doesn't add `.txt` to the end

---

## 6. Authenticate with Yahoo

Run this command to log in to Yahoo:

**Mac:**
```
python3 -c "from fantasy_baseball.auth.yahoo_auth import get_yahoo_session; get_yahoo_session()"
```

**Windows:**
```
python -c "from fantasy_baseball.auth.yahoo_auth import get_yahoo_session; get_yahoo_session()"
```

Your browser will open to a Yahoo login page. Log in with **your own Yahoo account** (the one connected to your fantasy league) and click **Allow** when prompted. The tool will save your login token so you don't have to do this every time.

---

## 7. Configure your league

Edit `config/league.yaml` to match your league's settings. Here's what to change:

```yaml
league:
  id: YOUR_LEAGUE_ID        # The number from your league URL
  num_teams: 12             # How many teams in your league
  game_code: mlb
  team_name: "Your Team Name"  # Your exact team name from Yahoo

draft:
  position: 1               # Your draft pick position

keepers: []                  # Add keepers if your league has them

roster_slots:                # Match your league's roster setup
  C: 1
  1B: 1
  2B: 1
  3B: 1
  SS: 1
  IF: 1
  OF: 4
  UTIL: 2
  P: 9
  BN: 2
  IL: 2

projections:
  systems:
    - steamer
    - zips
    - atc
  weights:
    steamer: 0.33
    zips: 0.33
    atc: 0.34
```

**Finding your league ID:** Go to your league page on Yahoo Fantasy. The URL looks like `baseball.fantasysports.yahoo.com/b2/12345` — the number at the end (e.g., `12345`) is your league ID.

---

## 8. Download projections

1. Go to https://www.fangraphs.com/projections
2. For each projection system (Steamer, ZiPS, ATC):
   - Select **Hitters**, click **Export Data** to download a CSV
   - Select **Pitchers**, click **Export Data** to download a CSV
3. Save all 6 CSV files into the `data/projections/` folder inside the `FantasyBaseball` directory
4. Keep the default FanGraphs filenames (they look like `fangraphs-leaderboard-projections-steamer-hitters.csv`)

---

## 9. Run it

**Draft simulator:**

Mac: `python3 scripts/simulate_draft.py`
Windows: `python scripts/simulate_draft.py`

**Lineup optimizer (once the season starts):**

Mac: `python3 scripts/run_lineup.py`
Windows: `python scripts/run_lineup.py`

---

## Troubleshooting

**"python is not recognized" (Windows):** Python wasn't added to PATH. Reinstall from python.org and check the "Add Python to PATH" box.

**"No module named fantasy_baseball":** You need to run the install command from Step 4. Make sure you're in the `FantasyBaseball` directory first (`cd FantasyBaseball`).

**"oauth.json not found":** Make sure the file is in the `config/` folder and the filename is exactly `oauth.json` (not `oauth.json.txt`).

**Yahoo login fails or says "unauthorized":** The Yahoo Developer app may need to be in production mode, or your Yahoo account may need to be added as a test user. Ask Hart.

**"Projections directory not found":** You need to download the FanGraphs CSVs (Step 8) and save them in `data/projections/`.
