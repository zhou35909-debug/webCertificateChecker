"""
Show random real URLs from the downloaded phishing dataset.
Run this AFTER train_model.py has been run at least once.

    cd backend
    python query_dataset.py           # show 10 random samples
    python query_dataset.py 20        # show 20 random samples
    python query_dataset.py 10 phish  # show only phishing URLs
    python query_dataset.py 10 safe   # show only safe URLs
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.data_source import sample_real_urls, load_training_data

def main():
    n      = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    filter_ = sys.argv[2].lower() if len(sys.argv) > 2 else "all"

    df = sample_real_urls(n=max(n * 10, 500))   # over-sample then filter

    if df.empty:
        print("No dataset cached. Running download now...")
        load_training_data()
        df = sample_real_urls(n=max(n * 10, 500))

    if df.empty:
        print("Could not load any URLs. Check your internet connection.")
        return

    if filter_ == "phish":
        df = df[df["label"] == 1]
    elif filter_ == "safe":
        df = df[df["label"] == 0]

    df = df.sample(min(n, len(df)), random_state=None).reset_index(drop=True)

    print(f"\n{'#':<4}  {'Label':<12}  URL")
    print("-" * 80)
    for i, row in df.iterrows():
        tag = "PHISHING" if row["label"] == 1 else "safe    "
        print(f"{i+1:<4}  {tag:<12}  {row['url']}")

    print()
    print("Tip: copy any phishing URL above into CertAgent to test the NN score.")
    print("     The rule-based engine checks the live SSL cert; the NN scores the URL structure.")

if __name__ == "__main__":
    main()
