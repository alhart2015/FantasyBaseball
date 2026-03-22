"""Worker: build board, run one strategy+scoring across all seeds, print results."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from simulate_draft import build_board_and_context, run_simulation

OPP = "1:three_closers,2:default,3:nonzero_sv,4:three_closers,5:nonzero_sv,6:nonzero_sv,7:default,9:nonzero_sv,10:three_closers"

strategy = sys.argv[1]
scoring = sys.argv[2]
seeds = [int(s) for s in sys.argv[3:]]

ctx = build_board_and_context()
for seed in seeds:
    r = run_simulation(ctx, strategy_name=strategy, scoring_mode=scoring,
                      adp_noise=15, seed=seed, opponent_strategies_str=OPP)
    print(f"RESULT:{strategy}:{scoring}:{seed}:{r['pts']}:{r['rank']}")
