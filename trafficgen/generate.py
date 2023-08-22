# -*- coding: utf-8 -*-
# @Author: TrafficGen Team + Shounak Ray
# @Date:   2023-08-14 15:07:03
# @Last Modified by:   Shounak Ray
# @Last Modified time: 2023-08-14 15:24:43

from trafficgen.traffic_generator.traffic_generator import TrafficGen
from trafficgen.traffic_generator.utils.utils import get_parsed_args
from trafficgen.utils.config import load_config_init

# Please keep this line here:
from trafficgen.utils.typedef import AgentType, RoadLineType, RoadEdgeType

if __name__ == "__main__":
    # Gets arguments from command line
    args = get_parsed_args()

    # Gets model configuration from .YAML file as specified by args.config
    cfg = load_config_init(args.config)
    print('loading checkpoint...')
    trafficgen = TrafficGen(cfg)
    print('Complete.\n')

    trafficgen.generate_scenarios(gif=args.gif, save_metadrive=args.save_metadrive)
