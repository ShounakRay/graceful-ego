from trafficgen.traffic_generator.traffic_generator import TrafficGen
from trafficgen.traffic_generator.utils.utils import get_parsed_args
from trafficgen.utils.config import load_config_init

# Please keep this line here:
from trafficgen.utils.typedef import AgentType, RoadLineType, RoadEdgeType
from pprint import pprint

if __name__ == "__main__":
    args = get_parsed_args()

    cfg = load_config_init(args.config)
    # NOTE: [SR] Added this line for clarity (confirm device assignment)
    pprint(f"Arguments: {cfg}")

    print("loading checkpoint...")
    trafficgen = TrafficGen(cfg)
    print("Complete.\n")

    trafficgen.generate_scenarios(gif=args.gif, save_metadrive=args.save_metadrive)
