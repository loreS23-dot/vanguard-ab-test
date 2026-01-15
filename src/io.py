from pathlib import Path
import pandas as pd


def load_raw(data_dir: str | Path):
    """
    Load raw Vanguard project datasets from a local folder.

    Expected files inside data_dir:
      - df_final_demo.txt
      - df_final_experiment_clients.txt
      - df_final_web_data_pt_1.txt
      - df_final_web_data_pt_2.txt
    """
    data_dir = Path(data_dir)

    demo = pd.read_csv(data_dir / "df_final_demo.txt")
    exp = pd.read_csv(data_dir / "df_final_experiment_clients.txt")
    w1 = pd.read_csv(data_dir / "df_final_web_data_pt_1.txt")
    w2 = pd.read_csv(data_dir / "df_final_web_data_pt_2.txt")

    return demo, exp, w1, w2
