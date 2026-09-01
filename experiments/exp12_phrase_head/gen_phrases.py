"""Deterministic phrase generator for the 184-class contrastive head.

Provenance-safe replacement for the hand-authored phrases.json: every phrase
is produced by one of four fixed template rules over a documented expansion
glossary. The glossary expands the dataset's abbreviated label names into the
terminology of the ROAD dataset paper (Singh et al., ROAD: The ROad event
Awareness Dataset, TPAMI 2022, Sec. 3: agents, actions, locations are defined
as ego-relative road-event constituents). No free-form wording: change the
glossary or a template and every phrase regenerates identically.

Rules:
  R1 agentness: fixed sentence.
  R2 agent:    "a {AGENT}"
  R3 action:   "something {ACTION}" (verb phrase); loc: "something {LOC}"
  R4 duplex:   "a {AGENT} {ACTION}"
     triplet:  "a {AGENT} {ACTION} {LOC}"

Usage: python3 gen_phrases.py [--out phrases_v2.json] [--diff phrases.json]
"""
import argparse, json
from pathlib import Path

E12 = Path(__file__).resolve().parent
JSON_PATH = "/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json"

# ---- expansion glossary (ROAD paper nomenclature; review with advisor) ----
AGENT = {
    "Ped": "pedestrian", "Car": "car", "Cyc": "cyclist", "Mobike": "motorbike",
    "SmalVeh": "small vehicle", "MedVeh": "medium vehicle", "LarVeh": "large vehicle",
    "Bus": "bus", "EmVeh": "emergency vehicle", "TL": "traffic light",
}
ACTION = {
    "Red": "showing the red stop signal", "Amber": "showing the amber caution signal",
    "Green": "showing the green go signal",
    "MovAway": "moving away from the ego vehicle", "MovTow": "moving towards the ego vehicle",
    "Mov": "moving", "Rev": "reversing", "Brake": "braking", "Stop": "stopped",
    "IncatLft": "indicating left", "IncatRht": "indicating right",
    "HazLit": "with hazard lights on", "TurLft": "turning left", "TurRht": "turning right",
    "MovRht": "moving to the right", "MovLft": "moving to the left", "Ovtak": "overtaking",
    "Wait2X": "waiting to cross", "XingFmLft": "crossing from the left",
    "XingFmRht": "crossing from the right", "Xing": "crossing", "PushObj": "pushing an object",
}
LOC = {
    "VehLane": "in the ego vehicle's lane", "OutgoLane": "in the outgoing lane",
    "OutgoCycLane": "in the outgoing cycle lane", "OutgoBusLane": "in the outgoing bus lane",
    "IncomLane": "in the incoming lane", "IncomCycLane": "in the incoming cycle lane",
    "IncomBusLane": "in the incoming bus lane", "Pav": "on the pavement",
    "LftPav": "on the left pavement", "RhtPav": "on the right pavement",
    "Jun": "at the junction", "xing": "at the pedestrian crossing",
    "BusStop": "at the bus stop", "parking": "in the parking area",
    "LftParking": "in the left parking area", "rightParking": "in the right parking area",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(E12 / "phrases_v2.json"))
    ap.add_argument("--diff", default=None)
    args = ap.parse_args()

    d = json.load(open(JSON_PATH))
    for name, table, labels in (("agent", AGENT, d["agent_labels"]),
                                ("action", ACTION, d["action_labels"]),
                                ("loc", LOC, d["loc_labels"])):
        missing = [l for l in labels if l not in table]
        assert not missing, f"glossary missing {name}: {missing}"

    out = {"agentness": "an object of interest in the road scene",
           "agent": {}, "action": {}, "loc": {}, "duplex": {}, "triplet": {}}
    for a in d["agent_labels"]:
        out["agent"][a] = f"a {AGENT[a]}"
    for c in d["action_labels"]:
        out["action"][c] = f"something {ACTION[c]}"
    for l in d["loc_labels"]:
        out["loc"][l] = f"something {LOC[l]}"
    for name in d["duplex_labels"]:
        a, c = name.split("-", 1)
        out["duplex"][name] = f"a {AGENT[a]} {ACTION[c]}"
    for name in d["triplet_labels"]:
        a, rest = name.split("-", 1)
        c, l = rest.rsplit("-", 1)
        out["triplet"][name] = f"a {AGENT[a]} {ACTION[c]} {LOC[l]}"

    Path(args.out).write_text(json.dumps(out, indent=1))
    n = 1 + sum(len(out[k]) for k in ("agent", "action", "loc", "duplex", "triplet"))
    print(f"[phrases] wrote {n} phrases -> {args.out}")

    if args.diff:
        old = json.load(open(args.diff))
        changed = 0
        for k in ("agent", "action", "loc", "duplex", "triplet"):
            for name, p in out[k].items():
                if old.get(k, {}).get(name) != p:
                    changed += 1
        print(f"[phrases] {changed}/{n-1} differ from {args.diff}")

if __name__ == "__main__":
    main()
