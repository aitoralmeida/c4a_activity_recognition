import json
import sys

# EAMs dir
DIR = './eams/'
# EAMs file
EAMS_FILE = DIR + 'eams.json'

def check_location(locations_a, locations_b):
    for location_a in locations_a:
        for location_b in locations_b:
            if location_a == location_b:
                return True
    return False

def check_start(starts_a, starts_b):
    return False

def transform_activity_dict_to_action_dict(activity_dict):
    action_dict = {}
    for activity, knowledge in activity_dict.items():
        for action in knowledge['actions']:
            key = action
            if key in action_dict:
                values = action_dict[key]
                locations = values['locations']
                start = values['start']
                locations.extend(knowledge['locations'])
                start.extend(knowledge['start'])
                action_dict[key] = values
            else:
                values = {}
                values['locations'] = knowledge['locations']
                values['start'] = knowledge['start']
                action_dict[key] = values
    return action_dict

def main(argv):
    # read EAMs from file
    with open(EAMS_FILE) as json_file:
        eams = json.load(json_file)
    # check EAMs struct
    print(eams)
    # transform activity knowledge to action knowledge
    action_dict = transform_activity_dict_to_action_dict(eams)
    # check new struct
    print(action_dict)
    # calculate edges of the graph
    edge_list = []
    for action, knowledge in action_dict.items():
        for another_action, another_knowledge in action_dict.items():
            if (action != another_action):
                # check locations correspondance
                if check_location(knowledge['locations'], another_knowledge['locations']):
                    edge_list.append([action, another_action])
                if check_start(knowledge['start'], another_knowledge['start']):
                    edge_list.append([action, another_action])
    # write graph edges to file
    with open("/results/actions.edgelist", "w") as edgelist_file:
        for edge in edge_list:
            edgelist_file.write(str(edge[0]) + " " + str(edge[1]) + "\n")

if __name__ == "__main__":
    main(sys.argv)