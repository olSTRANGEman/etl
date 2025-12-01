import pandas as pd
from scipy.io import arff
import argparse

def arff_to_csv(input_path, output_path):
    # Load ARFF
    data, meta = arff.loadarff(input_path)

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)

    # Convert byte strings to normal strings (ARFF categorical fields)
    df = df.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Save CSV
    df.to_csv(output_path, index=False)
    print(f"✓ Saved CSV to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to .arff file")
    parser.add_argument("output", help="Path to output .csv file")
    args = parser.parse_args()

    arff_to_csv(args.input, args.output)