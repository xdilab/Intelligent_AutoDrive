"""Version 1: canonical wording plus four compositional, fixed descriptions."""
import argparse, json
from common import INPUTS, read, save

def generate():
    ph=read(INPUTS['phrases']);labels=read(INPUTS['frames'])['labels']
    # Use the established vocabulary, while making implicit ego-relative relations explicit.
    agent=dict(ph['agent']);action=dict(ph['action']);loc=dict(ph['loc'])
    action.update(Brake='braking',Ovtak='overtaking',HazLit='showing hazard lights',
                  MovLft='moving to the left relative to the ego vehicle',
                  MovRht='moving to the right relative to the ego vehicle',
                  XingFmLft='crossing from the left relative to the ego vehicle',
                  XingFmRht='crossing from the right relative to the ego vehicle')
    loc.update(LftParking='in the parking area on the left of the ego vehicle',
               rightParking='in the parking area on the right of the ego vehicle',
               LftPav='on the pavement on the left of the ego vehicle',
               RhtPav='on the pavement on the right of the ego vehicle',
               OutgoLane='in a traffic lane going in the ego vehicle\'s direction',
               IncomLane='in a traffic lane coming toward the ego vehicle',
               OutgoCycLane='in a cycle lane going in the ego vehicle\'s direction',
               IncomCycLane='in a cycle lane coming toward the ego vehicle',
               OutgoBusLane='in a bus lane going in the ego vehicle\'s direction',
               IncomBusLane='in a bus lane coming toward the ego vehicle')
    rows=[dict(row=0,id='agentness',special=True,descriptions=[ph['agentness']])]
    for group in ['agent','action','loc','duplex','triplet']:
        for label in labels[group]:
            canonical=ph[group][label]
            if group=='agent':
                a=agent[label]
                extra=[f'The road agent is {a}.',f'There is {a}.',f'This is {a}.',f'The scene contains {a}.']
            elif group=='action':
                v=action[label]
                subject='A traffic light' if label in ['Red','Amber','Green'] else 'An agent'
                extra=[f'{subject} is {v}.',f'The observed state is {v}.',f'The scene shows an agent {v}.',f'This road event involves an agent {v}.']
            elif group=='loc':
                l=loc[label]
                extra=[f'An agent is {l}.',f'The agent is located {l}.',f'The scene contains an agent {l}.',f'This road event involves an agent {l}.']
            else:
                parts=label.split('-');a=agent[parts[0]];v=action[parts[1]]
                l=(' '+loc[parts[2]]) if group=='triplet' else ''
                extra=[f'{a.capitalize()} is {v}{l}.',f'The scene shows {a} {v}{l}.',f'This road event involves {a} {v}{l}.',f'There is {a} {v}{l}.']
            descriptions=[canonical]+extra
            assert len(set(descriptions))==5,(group,label,descriptions)
            rows.append(dict(row=len(rows),id=f'{group}:{label}',special=False,descriptions=descriptions))
    assert len(rows)==184 and sum(len(r['descriptions']) for r in rows)==916
    return dict(version='comp765-description-templates-v1',rows=rows,
                glossary=dict(agent=agent,action=action,loc=loc),
                rules='Canonical exact, then four fixed observation/existence/event templates. No stochastic generation.',
                caveats=['Canonical wording is immutable, including historical Brake brake-light wording and parking phrases that say parked. Added descriptions use only annotated state/location. This inherited wording discrepancy is disclosed; it is not silently corrected in the baseline.',
                         'Waiting to cross is an observed dataset action; no future crossing outcome is added.',
                         'Outgoing/incoming and left/right locations are ego-relative; traffic-light colors remain displayed signal states, not agent motion.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();save(a.out,generate())
