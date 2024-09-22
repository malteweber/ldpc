import argparse
import json

from utils.graph_utils import girth_average
from utils.simulation_utils import test_girth_metric_binary_symmetric

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Add arguments: one positional and one optional
    parser.add_argument("--nr_of_matrices", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--w_r", type=int)
    parser.add_argument("--w_c", type=int)
    parser.add_argument("--nr_of_transmissions", type=int)
    parser.add_argument("--bp_iterations", type=int)
    parser.add_argument("--flip_rates", type=float, nargs="+")
    parser.add_argument("--output", type=str)


    # Parse the arguments
    args = parser.parse_args()

    print(args.flip_rates)

    c, e = test_girth_metric_binary_symmetric(
        girth_metric=girth_average,
        nr_of_matrices=args.nr_of_matrices,
        n=args.n,
        w_r=args.w_r,
        w_c=args.w_c,
        nr_of_transmissions=args.nr_of_transmissions,
        bp_iterations=args.bp_iterations,
        flip_rates=args.flip_rates)

    result_dict = {
        "convergence_rates": c,
        "bit_error_rates": e,
        "parameters": vars(args)
    }

    with open(args.output, "w") as f:
        f.write(json.dumps(result_dict))

