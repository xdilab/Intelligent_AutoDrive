Subject: ROAD++ progress update and thesis proposal defense scheduling

Dear Dr. Moradi,

I wanted to share a brief update on my progress with the ROAD++ project and also ask about scheduling my thesis proposal defense before the end of the semester.

So far, I have been focusing on understanding the ROAD-Waymo dataset structure, annotation scheme, and the logic-related constraints implicit in its compositional labels. In particular, the dataset does not only provide separate agent, action, and location labels, but also valid duplex and triplet combinations that indicate which scene interpretations are semantically consistent. I think this is useful because it provides a principled way to guide learning beyond standard supervision and may help connect structured prediction with interpretable text-based reasoning about the scene. I also reviewed the existing ROAD-Waymo neuro-symbolic baseline and confirmed that the paper already applies t-norm based constraint loss in the 3D-RetinaNet setting, so I am treating that primarily as a baseline replication rather than as a novel contribution.

For the research direction, I have been thinking about model architectures that better align with your interest in text-based reasoning about the scene. The most promising idea at the moment is a generative vision-language model that produces both structured scene labels and natural language reasoning, while applying logic-based constraints to the structured outputs during training. My goal would be for the model to generate explanations such as pedestrian or driver intent in a way that remains consistent with the scene semantics encoded in ROAD-Waymo.

In parallel, I have also been looking into JEPA-style world models as a possible thesis direction. In particular, V-JEPA and VL-JEPA seem relevant because they model spatiotemporal scene dynamics at a higher level of abstraction, while LeWM is especially interesting because it appears feasible to run on a regular workstation. My thought is that these models could be used as feature extractors or latent predictive backbones for downstream tasks, so that prediction is performed on learned scene representations rather than only on raw pixels. This seems closer to how humans reason about scenes, in the sense that the model would focus more on objects, motion, and scene evolution at an abstract level. LeWM is especially appealing as an exploratory direction because it may make it possible to study future-state prediction and intent-related reasoning without requiring very large-scale compute, while VL-JEPA may be better aligned with the longer-term goal of producing text-based scene reasoning.

I also came across recent work on structured 4D scene understanding for VLM reasoning, such as SNOW, which is not a JEPA model but still reinforces the idea that higher-level spatiotemporal world representations may be useful for grounded scene reasoning and language output. This seems relevant because it suggests that explicit scene structure and temporal grounding could be beneficial even outside strictly JEPA-based approaches.

At the implementation level, I have already organized the possible approaches and started baseline setup and training work so that I can establish a comparison point before moving into the more novel architecture. My current plan is to use the baseline as a reference and then develop the thesis contribution around constrained language-based scene reasoning, possibly with a JEPA- or world-model-inspired backbone if that seems appropriate.

As a next step, I will also be training a VLM-based architecture for this direction, and I will try to have preliminary results by Thursday so that I can better assess feasibility and share something more concrete.

Since the project direction is becoming clearer, I would also like to move forward with my thesis proposal defense as soon as possible. If possible, I would really appreciate scheduling a date before the semester ends. Please let me know what dates might work for you, and I will adjust my schedule accordingly.

Best regards,

[Your Name]
